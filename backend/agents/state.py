from dataclasses import dataclass, field


@dataclass
class AgentState:
    video_id: str
    goal: str
    history: list = field(default_factory=list)       # prior chat turns [{"role","content"}]
    observations: list = field(default_factory=list)  # tool outputs
    tools_tried: set = field(default_factory=set)     # dedupe: (tool, args-json)
    seen_urls: list = field(default_factory=list)     # URLs returned by web_search
    failure_reason: str = ""                          # written by verifier, read by planner
    iteration: int = 0
    max_iterations: int = 3
    trace: list = field(default_factory=list)         # debug log returned to the client
