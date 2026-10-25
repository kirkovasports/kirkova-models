"""MLflow model registry wrapper.

MLflow is the source of truth for versioned artifacts.
The `models` DB table is a mirror for SQL joins on predictions.
"""

from __future__ import annotations

import contextlib
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.db.models import ModelRecord

import mlflow
import mlflow.lightgbm
import mlflow.sklearn
import mlflow.xgboost

from src.core.config import settings
from src.core.logging import get_logger

log = get_logger(__name__)


def _setup_mlflow() -> None:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)


def log_model_run(
    run_name: str,
    sport: str,
    kind: str,
    target: str,
    model: Any,
    metrics: dict[str, float],
    params: dict[str, Any],
    feature_names: list[str],
    training_range: tuple[str, str],
    model_framework: str = "sklearn",
) -> str:
    """Log a training run to MLflow. Returns the run_id."""
    _setup_mlflow()

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(
            {
                "sport": sport,
                "kind": kind,
                "target": target,
                "training_start": training_range[0],
                "training_end": training_range[1],
                "n_features": len(feature_names),
            }
        )
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_dict({"feature_names": feature_names}, "feature_names.json")

        if model_framework == "xgboost":
            mlflow.xgboost.log_model(model, artifact_path="model")
        elif model_framework == "lightgbm":
            mlflow.lightgbm.log_model(model, artifact_path="model")
        else:
            # StackedEnsemble (winner model) and QuantileBundle (props model) both
            # wrap CatBoost/LightGBM/XGBoost sub-models, none of which are in
            # skops' default trusted-type allowlist. These are our own
            # application types (or well-known ML libraries), not arbitrary user
            # input, so explicitly trusting them is safe.
            #
            # LGBMRegressor/QuantileBundle were missing here until 2026-07-10 -
            # every train_props run was completing training successfully and
            # then silently failing at this exact save step for months, so no
            # prop model had actually been promoted despite training "succeeding"
            # (log_model_run's caller sees the exception, not a partial save -
            # train_props.py's per-stat try/except just logged it as
            # 'train_props.failed' and moved to the next stat).
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                skops_trusted_types=[
                    "catboost.core.CatBoostClassifier",
                    "collections.OrderedDict",
                    "lightgbm.basic.Booster",
                    "lightgbm.sklearn.LGBMClassifier",
                    "lightgbm.sklearn.LGBMRegressor",
                    "numpy.dtype",
                    "sklearn.impute._base.SimpleImputer",
                    "sklearn.isotonic.IsotonicRegression",
                    "sklearn.pipeline.Pipeline",
                    "src.models.train_props.OverUnderHead",
                    "src.models.train_props.QuantileBundle",
                    "src.models.train_efl_winner.EflLineAnchoredModel",
                    "src.models.train_efl_winner.FallbackModel",
                    "src.models.train_winner.LineAnchoredModel",
                    "src.models.train_winner.StackedEnsemble",
                    "xgboost.core.Booster",
                    "xgboost.sklearn.XGBClassifier",
                ],
            )

        log.info(
            "mlflow.run_logged",
            run_id=run.info.run_id,
            sport=sport,
            kind=kind,
            target=target,
            metrics=metrics,
        )
        return run.info.run_id  # type: ignore[no-any-return]


def _cap_inference_threads(model: Any) -> Any:
    """Force every LightGBM/CatBoost/XGBoost sub-estimator in a loaded model to
    single-threaded inference.

    train_winner.py/train_props.py deliberately set n_jobs=16/thread_count=16
    on LightGBM/CatBoost so training runs fast on KS-6-B's dedicated,
    concurrency=1 worker-training container - but those values are baked into
    the pickled model artifact and travel with it into scoring. The
    operational `worker` container runs at concurrency=8 (Celery prefork), so
    a single predict_proba() call was spinning up a fresh 16-thread OpenMP
    team *inside an already-forked child process* on every game scored.
    OpenMP's runtime isn't fork-safe once a multi-threaded team exists post-
    fork - confirmed 2026-07-25 via prod `dmesg`: this crashed with SIGSEGV
    inside libgomp.so on essentially every scoring tick, silently losing the
    task (WorkerLostError - no exception the task code ever saw, no
    predictions written, for any sport). Capping to 1 thread here means no
    OpenMP team is ever created, which sidesteps the bug entirely - scoring
    one row at a time never needed 16 threads anyway.
    """
    seen: set[int] = set()

    def _cap(obj: Any) -> None:
        if obj is None or id(obj) in seen:
            return
        seen.add(id(obj))
        get_params = getattr(obj, "get_params", None)
        if callable(get_params):
            try:
                params = get_params()
            except Exception:
                params = {}
            overrides = {p: 1 for p in ("n_jobs", "thread_count") if p in params}
            if overrides:
                with contextlib.suppress(Exception):
                    obj.set_params(**overrides)
        # XGBClassifier.set_params(n_jobs=1) only affects predict_proba() -
        # score.py's SHAP-contribs path calls .get_booster().predict(...)
        # directly on the raw native Booster, which keeps its own nthread
        # baked in from training regardless of the sklearn wrapper's n_jobs.
        # Confirmed 2026-07-25: MLB kept segfaulting after the wrapper-level
        # cap alone because this path was still untouched.
        get_booster = getattr(obj, "get_booster", None)
        if callable(get_booster):
            with contextlib.suppress(Exception):
                get_booster().set_param({"nthread": 1})
        for attr in ("xgb_clf", "lgb_clf", "cat_clf", "fallback"):
            _cap(getattr(obj, attr, None))
        quantile_models = getattr(obj, "quantile_models", None)
        if isinstance(quantile_models, dict):
            for sub in quantile_models.values():
                _cap(sub)

    _cap(model)
    return model


def load_model(run_id: str, framework: str = "sklearn", *, retries: int = 3) -> Any:
    """Load a model artifact from MLflow by run_id.

    Retries on failure: skops/xgboost's GPU device probe on load has been
    observed to intermittently throw a spurious ModuleNotFoundError (seen as
    "No module named 'no_Cryptodome'") when it races GPU visibility checks
    against another process on the host - promote_model already smoke-tests
    that the artifact itself loads before promoting, so a load failure here
    is a transient runtime hiccup, not a corrupt artifact (see 2026-07-14
    incident: this silently zeroed out WNBA picks for a day when it hit both
    of that day's games in one scoring run).
    """
    _setup_mlflow()
    uri = f"runs:/{run_id}/model"
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            if framework == "xgboost":
                return mlflow.xgboost.load_model(uri)
            elif framework == "lightgbm":
                return mlflow.lightgbm.load_model(uri)
            return _cap_inference_threads(mlflow.sklearn.load_model(uri))
        except Exception as exc:
            last_exc = exc
            if attempt < retries - 1:
                log.warning(
                    "model.load_retry",
                    run_id=run_id,
                    attempt=attempt + 1,
                    error=str(exc),
                )
                time.sleep(1.5 * (attempt + 1))
    raise last_exc  # type: ignore[misc]


def get_run_metrics(run_id: str) -> dict[str, float]:
    _setup_mlflow()
    client = mlflow.MlflowClient()
    run = client.get_run(run_id)
    return dict(run.data.metrics)


def promote_model(
    session: Any,  # SQLAlchemy Session
    run_id: str,
    sport_id: int,
    kind: str,
    target: str,
    version: str,
    metrics: dict[str, float],
    feature_spec_hash: str,
    model_framework: str = "sklearn",
) -> ModelRecord:
    """Deactivate old active model for this (sport, kind, target), activate new one.

    `model_framework` is accepted for the soccer promotion tasks (src/tasks/train_tasks.py passes "xgboost" for the top-5 / MLS slots): they always
    did, and the function had no such parameter, so the first soccer retrain that beat its champion would have raised TypeError instead of promoting.
    The smoke test below loads through mlflow.pyfunc, which reads the flavour from the artifact itself, so the value does not change behaviour.

    Smoke-tests the artifact loads before touching the DB - a run can have a
    great recorded metric yet be unloadable (e.g. a later code refactor
    renamed/removed a class its pickle depends on). Promoting an unloadable
    model would silently zero out live scoring until someone notices predictions
    stopped (see 2026-07-10 incident: a 6-week-old run got re-promoted by
    evaluate_and_promote's un-windowed "best logloss ever" search and broke
    MLB scoring for hours before detection).
    """
    from src.core.time import utc_now
    from src.db.models import ModelRecord

    _setup_mlflow()
    try:
        mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
    except Exception as exc:
        log.error("model.promote_smoke_test_failed", run_id=run_id, error=str(exc))
        raise RuntimeError(
            f"Refusing to promote run {run_id}: model failed to load ({exc})"
        ) from exc

    # Deactivate existing
    session.query(ModelRecord).filter_by(
        sport_id=sport_id, kind=kind, target=target, active=True
    ).update({"active": False})

    record = ModelRecord(
        sport_id=sport_id,
        kind=kind,
        target=target,
        version=version,
        mlflow_run_id=run_id,
        trained_at=utc_now(),
        active=True,
        metrics=metrics,
        feature_spec_hash=feature_spec_hash,
    )
    session.add(record)
    session.flush()
    log.info("model.promoted", sport_id=sport_id, kind=kind, target=target, version=version)
    return record


def rollback_model(session: Any, sport_id: int, kind: str, target: str) -> bool:
    """Revert to the previous active model version."""
    from src.db.models import ModelRecord

    models = (
        session.query(ModelRecord)
        .filter_by(sport_id=sport_id, kind=kind, target=target)
        .order_by(ModelRecord.trained_at.desc())
        .limit(2)
        .all()
    )

    if len(models) < 2:
        log.warning("model.rollback_impossible", sport_id=sport_id, kind=kind, target=target)
        return False

    current, previous = models[0], models[1]
    current.active = False
    previous.active = True
    session.flush()
    log.info("model.rolled_back", to_version=previous.version)
    return True
