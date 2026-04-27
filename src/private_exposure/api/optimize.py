from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from private_exposure.api.deps import get_session
from private_exposure.schemas.optimize import OptimizeRequest, OptimizeResponse
from private_exposure.services.optimize_service import run_optimizer

router = APIRouter(prefix="/optimize", tags=["optimize"])

@router.post("", response_model=OptimizeResponse)
def optimize(req: OptimizeRequest, session: Session = Depends(get_session)):
    try:
        return run_optimizer(session, req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))