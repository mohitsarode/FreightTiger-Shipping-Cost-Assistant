import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List

@dataclass
class CallRecord:
    timestamp: str
    call_type: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    model_name: str

@dataclass
class TokenTracker:
    model_name: str = "gemini-1.5-flash / local-deterministic"
    input_rate_per_million: float = 0.075    # $0.075 / 1M input tokens
    output_rate_per_million: float = 0.300   # $0.300 / 1M output tokens
    
    total_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    history: List[Dict[str, Any]] = field(default_factory=list)
    
    def log_call(self, input_text: str, output_text: str, call_type: str = "explanation_generation") -> Dict[str, Any]:
        """
        Estimates tokens (using standard ~4 chars/token heuristic or exact word tokens)
        and computes costs according to published model rates.
        """
        # Standard conservative token estimation: ~4 chars per token + word-based bounds
        in_tokens = max(1, len(input_text.split()), int(len(input_text) / 3.8))
        out_tokens = max(1, len(output_text.split()), int(len(output_text) / 3.8))
        
        call_cost = (in_tokens / 1_000_000.0 * self.input_rate_per_million) + \
                    (out_tokens / 1_000_000.0 * self.output_rate_per_million)
                    
        self.total_calls += 1
        self.total_input_tokens += in_tokens
        self.total_output_tokens += out_tokens
        self.total_cost_usd += call_cost
        
        record = {
            "call_index": self.total_calls,
            "call_type": call_type,
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
            "cost_usd": round(call_cost, 7),
            "model": self.model_name
        }
        self.history.append(record)
        return record
        
    def get_summary(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "total_calls": self.total_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "estimated_cost_usd": round(self.total_cost_usd, 6),
            "published_pricing": {
                "input_per_1m_tokens_usd": self.input_rate_per_million,
                "output_per_1m_tokens_usd": self.output_rate_per_million
            }
        }
        
    def save_log(self, output_path: Path):
        data = self.get_summary()
        data["call_history"] = self.history
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
