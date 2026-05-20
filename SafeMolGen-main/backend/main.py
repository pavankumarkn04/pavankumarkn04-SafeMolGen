"""FastAPI application for SafeMolGen-DrugOracle."""
import asyncio
import json
import queue
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models.generator.safemolgen import SafeMolGen
from utils.chemistry import validate_smiles, calculate_properties
from utils.condition_vector import get_target_condition
from backend.pipeline_loader import load_pipeline
from backend.molecule_svg import draw_molecule_2d, normalize_smiles_for_display


# --- Pydantic schemas ---

class GenerateRequest(BaseModel):
    n: int = 20
    temperature: float = 0.8
    top_k: int = 40
    logp_min: Optional[float] = None
    logp_max: Optional[float] = None
    mw_min: Optional[float] = None
    mw_max: Optional[float] = None
    hbd_max: Optional[int] = None
    hba_max: Optional[int] = None
    tpsa_max: Optional[float] = None
    qed_min: Optional[float] = None
    target_phase: Optional[float] = None


class AnalyzeRequest(BaseModel):
    smiles: str


class CompareRequest(BaseModel):
    smiles_list: List[str] = Field(..., min_length=2)


class DesignRequest(BaseModel):
    target_success: float = 0.3
    max_iterations: int = 10
    candidates_per_iteration: int = 250
    top_k: int = 40
    safety_threshold: float = 0.2
    require_no_structural_alerts: bool = False
    property_targets: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Molecular/ADMET targets: logp (range), mw, hbd, hba, tpsa, qed, solubility, ppbr, clearance_hepatocyte_max",
    )
    seed_smiles: Optional[str] = None
    use_rl_model: bool = False
    selection_mode: str = "phase_weighted"
    diversity_tanimoto_max: float = 0.7
    n_restarts: int = 0
    design_mode: str = "single"
    use_reranker: bool = False
    reranker_top_k: int = 200
    population_size: int = 20
    generations: int = 10
    ensure_target: bool = False
    exploration_fraction: float = 0.25
    use_oracle_feedback: bool = True
    use_phase_aware_steering: Optional[bool] = None
    first_iteration_temperature: Optional[float] = None
    use_improvement_pacing: Optional[bool] = None
    max_step_per_iteration: Optional[float] = None
    max_rotatable_bonds: int = 15


# --- App setup ---

_generator: Optional[SafeMolGen] = None


def _get_checkpoint_path() -> Path:
    import os
    env_path = os.environ.get("GENERATOR_CHECKPOINT")
    if env_path and Path(env_path).exists():
        return Path(env_path)
    default = _PROJECT_ROOT / "checkpoints" / "generator"
    return default


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _generator
    path = _get_checkpoint_path()
    if (path / "model.pt").exists() and (path / "tokenizer.json").exists():
        try:
            _generator = SafeMolGen.from_pretrained(str(path), device="cpu")
        except Exception:
            _generator = None
    app.state.pipeline = None
    app.state.pipeline_rl = None
    yield
    app.state.pipeline = None
    app.state.pipeline_rl = None


app = FastAPI(
    title="SafeMolGen-DrugOracle API",
    version="1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_pipeline(use_rl_model: bool = False):
    key = "pipeline_rl" if use_rl_model else "pipeline"
    p = getattr(app.state, key, None)
    if p is None:
        try:
            p = load_pipeline(use_rl_model=use_rl_model)
            setattr(app.state, key, p)
        except Exception as e:
            import logging
            logging.warning(f"Failed to load pipeline: {e}")
            p = None
    return p


# --- Routes ---

@app.get("/api/v1/health")
def health():
    try:
        cond_dim = getattr(_generator.model, "cond_dim", 0) if _generator else 0
        pipeline = _get_pipeline(use_rl_model=False)
        return {
            "status": "ok",
            "generator_loaded": _generator is not None,
            "conditioning_available": cond_dim > 0,
            "models_loaded": pipeline is not None,
        }
    except Exception as e:
        return {
            "status": "ok",
            "generator_loaded": _generator is not None,
            "conditioning_available": False,
            "models_loaded": False,
            "warning": "Models not available"
        }


@app.get("/api/v1/config")
def config():
    try:
        pipeline = _get_pipeline(use_rl_model=False)
        has_reranker = pipeline is not None and getattr(pipeline, "reranker", None) is not None
        gen_early_available = pipeline is not None and getattr(pipeline, "generator_early", None) is not None
    except Exception:
        pipeline = None
        has_reranker = False
        gen_early_available = False
    return {
        "max_iterations_min": 1,
        "max_iterations_max": 50,
        "target_success_min": 0.1,
        "target_success_max": 0.95,
        "top_k_min": 1,
        "top_k_max": 80,
        "selection_modes": ["overall", "pareto", "diversity", "phase_weighted", "bottleneck"],
        "design_modes": ["single", "restarts", "evolutionary"],
        "has_reranker": has_reranker,
        "diversity_tanimoto_max_default": 0.7,
        "exploration_fraction_default": 0.25,
        "first_iteration_temperature_default": 1.4,
        "generator_early_available": gen_early_available,
        "default_property_targets": {
            "logp": [2.0, 5.0],
            "mw_min": 150,
            "mw": 500,
            "hbd": 5,
            "hba": 10,
            "tpsa": 140.0,
            "qed": 0.5,
        },
    }


def _passes_filters(props: Optional[Dict[str, Any]], req: GenerateRequest) -> bool:
    if props is None:
        return False
    if req.logp_min is not None and props.get("logp", 0) < req.logp_min:
        return False
    if req.logp_max is not None and props.get("logp", 0) > req.logp_max:
        return False
    if req.mw_min is not None and props.get("mw", 0) < req.mw_min:
        return False
    if req.mw_max is not None and props.get("mw", 0) > req.mw_max:
        return False
    if req.hbd_max is not None and props.get("hbd", 0) > req.hbd_max:
        return False
    if req.hba_max is not None and props.get("hba", 0) > req.hba_max:
        return False
    if req.tpsa_max is not None and props.get("tpsa", 0) > req.tpsa_max:
        return False
    if req.qed_min is not None and props.get("qed", 0) < req.qed_min:
        return False
    return True


def _has_property_filters(req: GenerateRequest) -> bool:
    return any((
        req.logp_min is not None, req.logp_max is not None,
        req.mw_min is not None, req.mw_max is not None,
        req.hbd_max is not None, req.hba_max is not None,
        req.tpsa_max is not None, req.qed_min is not None,
    ))


@app.post("/api/v1/generate")
def generate(req: GenerateRequest):
    if _generator is None:
        raise HTTPException(status_code=503, detail="Generator not loaded. Set GENERATOR_CHECKPOINT or add checkpoints/generator.")
    condition = None
    cond_dim = getattr(_generator.model, "cond_dim", 0)
    if cond_dim > 0:
        phase = req.target_phase
        if phase is None:
            phase = 0.6 if _has_property_filters(req) else 0.5
        phase = max(0.5, min(0.75, float(phase)))
        condition = get_target_condition(device="cpu", phase=phase)
    samples = _generator.generate(
        n=req.n, temperature=req.temperature, top_k=req.top_k,
        device="cpu", condition=condition,
    )
    results: List[Dict[str, Any]] = []
    passed = 0
    valid_count = 0
    for smi in samples:
        valid = validate_smiles(smi)
        if valid:
            valid_count += 1
        props = calculate_properties(smi) if valid else None
        passes = _passes_filters(props, req)
        if passes:
            passed += 1
        results.append({
            "smiles": smi, "valid": valid, "passed_filters": passes,
            "logp": round(props["logp"], 2) if props else None,
            "mw": round(props["mw"], 1) if props else None,
            "hbd": props.get("hbd") if props else None,
            "hba": props.get("hba") if props else None,
            "tpsa": round(props["tpsa"], 1) if props else None,
            "qed": round(props["qed"], 3) if props else None,
        })
    return {
        "results": results,
        "summary": {"total": len(samples), "valid": valid_count, "passed_filters": passed},
    }


@app.get("/api/v1/molecule/svg")
def molecule_svg(smiles: str, width: int = 400, height: int = 300):
    if not smiles or not smiles.strip():
        raise HTTPException(status_code=400, detail="smiles required")
    svg = draw_molecule_2d(smiles.strip(), size=(width, height))
    if svg is None:
        raise HTTPException(status_code=400, detail="Invalid SMILES")
    return Response(content=svg, media_type="image/svg+xml")


@app.post("/api/v1/analyze")
def analyze(req: AnalyzeRequest):
    pipeline = _get_pipeline(use_rl_model=False)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Train models first.")
    smiles = req.smiles.strip()
    if not smiles:
        raise HTTPException(status_code=400, detail="smiles required")
    prediction = pipeline.evaluate_molecule(smiles)
    if prediction is None:
        raise HTTPException(status_code=400, detail="Invalid SMILES")
    return {"smiles": smiles, "prediction": prediction.to_dict()}


@app.post("/api/v1/compare")
def compare(req: CompareRequest):
    pipeline = _get_pipeline(use_rl_model=False)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Train models first.")
    smiles_list = [s.strip() for s in req.smiles_list if s.strip()]
    if len(smiles_list) < 2:
        raise HTTPException(status_code=400, detail="At least 2 SMILES required")
    return {"results": pipeline.compare_molecules(smiles_list)}


def _design_kw(req: DesignRequest):
    seed = (req.seed_smiles or "").strip() or None
    candidates_per_iteration = req.candidates_per_iteration
    if req.property_targets:
        candidates_per_iteration = max(candidates_per_iteration, 350)
    kw = {
        "target_success": req.target_success,
        "max_iterations": req.max_iterations,
        "candidates_per_iteration": candidates_per_iteration,
        "top_k": req.top_k,
        "safety_threshold": req.safety_threshold,
        "require_no_structural_alerts": req.require_no_structural_alerts,
        "property_targets": req.property_targets,
        "seed_smiles": seed,
        "use_oracle_feedback": req.use_oracle_feedback,
        "show_progress": False,
        "on_iteration_done": None,
        "selection_mode": (req.selection_mode or "phase_weighted").lower(),
        "diversity_tanimoto_max": req.diversity_tanimoto_max,
        "use_reranker": req.use_reranker,
        "reranker_top_k": req.reranker_top_k,
    }
    kw["exploration_fraction"] = req.exploration_fraction
    if req.use_phase_aware_steering is not None:
        kw["use_phase_aware_steering"] = req.use_phase_aware_steering
    if req.first_iteration_temperature is not None:
        kw["first_iteration_temperature"] = req.first_iteration_temperature
    if req.use_improvement_pacing is not None:
        kw["use_improvement_pacing"] = req.use_improvement_pacing
    if req.max_step_per_iteration is not None:
        kw["max_step_per_iteration"] = req.max_step_per_iteration
    return kw


def _run_one_design(pipeline, req: DesignRequest, kw: dict) -> tuple:
    design_mode = getattr(req, "design_mode", "single")
    n_restarts = getattr(req, "n_restarts", 0) or (5 if design_mode == "restarts" else 0)
    if design_mode == "evolutionary":
        result = pipeline.design_molecule_evolutionary(
            population_size=getattr(req, "population_size", 20),
            generations=getattr(req, "generations", 10),
            target_success=req.target_success,
            show_progress=False,
        )
        return result, "evolutionary"
    if n_restarts > 0:
        result = pipeline.design_molecule_with_restarts(
            n_restarts=n_restarts, show_progress=False, **kw,
        )
        return result, "restarts"
    result = pipeline.design_molecule(**kw)
    return result, "single"


@app.post("/api/v1/design")
def design_sync(req: DesignRequest):
    pipeline = _get_pipeline(use_rl_model=req.use_rl_model)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Train models first.")
    kw = _design_kw(req)

    if getattr(req, "ensure_target", False):
        best_result = None
        best_strategy = None
        result = pipeline.design_molecule(**kw)
        best_result, best_strategy = result, "single"
        if not result.target_achieved:
            result = pipeline.design_molecule_with_restarts(n_restarts=5, show_progress=False, **kw)
            if result.final_prediction.overall_prob > best_result.final_prediction.overall_prob:
                best_result, best_strategy = result, "restarts_5"
            if not best_result.target_achieved:
                result = pipeline.design_molecule_with_restarts(
                    n_restarts=5, show_progress=False, **{**kw, "selection_mode": "diversity"},
                )
                if result.final_prediction.overall_prob > best_result.final_prediction.overall_prob:
                    best_result, best_strategy = result, "all_solutions"
        result = best_result
        out = result.to_dict()
        out["recommendations"] = result.final_prediction.recommendations
        out["_strategy_used"] = best_strategy
        if result.final_smiles:
            out["canonical_smiles"] = normalize_smiles_for_display(result.final_smiles)
        return out

    result, strategy = _run_one_design(pipeline, req, kw)
    out = result.to_dict()
    out["recommendations"] = result.final_prediction.recommendations
    out["_strategy_used"] = strategy
    if result.final_smiles:
        out["canonical_smiles"] = normalize_smiles_for_display(result.final_smiles)
    return out


def _sse_message(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


@app.post("/api/v1/design/stream")
async def design_stream(req: DesignRequest):
    pipeline = _get_pipeline(use_rl_model=req.use_rl_model)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Train models first.")

    q: queue.Queue = queue.Queue()

    def on_iteration_done(partial_result):
        q.put(("iteration", partial_result.to_dict()))

    def run_design():
        try:
            kw = _design_kw(req)
            kw["on_iteration_done"] = on_iteration_done
            if getattr(req, "ensure_target", False):
                best_result = None
                best_strategy = None
                result = pipeline.design_molecule(**kw)
                best_result, best_strategy = result, "single"
                if not result.target_achieved:
                    kw_no_cb = {k: v for k, v in kw.items() if k != "on_iteration_done"}
                    result = pipeline.design_molecule_with_restarts(n_restarts=5, show_progress=False, **kw_no_cb)
                    if result.final_prediction.overall_prob > best_result.final_prediction.overall_prob:
                        best_result, best_strategy = result, "restarts_5"
                    if not best_result.target_achieved:
                        result = pipeline.design_molecule_with_restarts(
                            n_restarts=5, show_progress=False, **{**kw_no_cb, "selection_mode": "diversity"},
                        )
                        if result.final_prediction.overall_prob > best_result.final_prediction.overall_prob:
                            best_result, best_strategy = result, "all_solutions"
                result = best_result
                out = result.to_dict()
                out["recommendations"] = result.final_prediction.recommendations
                out["_strategy_used"] = best_strategy
                if result.final_smiles:
                    out["canonical_smiles"] = normalize_smiles_for_display(result.final_smiles)
                q.put(("done", out))
            else:
                result, strategy = _run_one_design(pipeline, req, kw)
                out = result.to_dict()
                out["recommendations"] = result.final_prediction.recommendations
                out["_strategy_used"] = strategy
                if result.final_smiles:
                    out["canonical_smiles"] = normalize_smiles_for_display(result.final_smiles)
                q.put(("done", out))
        except Exception as e:
            q.put(("error", {"detail": str(e)}))

    thread = threading.Thread(target=run_design)
    thread.start()

    async def event_generator():
        yield _sse_message({"event": "started", "data": {"message": "Pipeline started. Loading models if needed..."}})
        loop = asyncio.get_event_loop()
        try:
            while True:
                kind, payload = await loop.run_in_executor(None, q.get)
                if kind == "iteration":
                    if payload.get("final_smiles"):
                        payload["canonical_smiles"] = normalize_smiles_for_display(payload["final_smiles"])
                    yield _sse_message({"event": "iteration", "data": payload})
                elif kind == "done":
                    yield _sse_message({"event": "done", "data": payload})
                    break
                elif kind == "error":
                    yield _sse_message({"event": "error", "data": payload})
                    break
        finally:
            thread.join(timeout=0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
