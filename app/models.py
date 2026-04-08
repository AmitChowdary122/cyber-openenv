# app/models.py
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class Observation(BaseModel):
    logs: List[str]
    alerts: List[str]
    system_state: Dict[str, Any]
    goal: Optional[str] = None
    # NEW: Reasoning signals for agent
    suspicion_scores: Optional[Dict[str, float]] = None      # IP -> suspicion (0.0-1.0)
    top_suspicious_ips: Optional[List[str]] = None           # Top 3 most suspicious IPs (by score)
    # NEW: investigation result
    investigation_result: Optional[Dict[str, Any]] = None    # IP -> enriched info

class Action(BaseModel):
    action_type: str  # e.g., "block_ip", "quarantine", "analyze", "ignore", "investigate_ip"
    parameters: Dict[str, Any] = {}

class Reward(BaseModel):
    value: float