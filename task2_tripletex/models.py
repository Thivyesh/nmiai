"""Dataclasses for Tripletex agent input/output mapping."""

from dataclasses import dataclass, field


@dataclass
class FileAttachment:
    filename: str
    content_base64: str
    mime_type: str


@dataclass
class TripletexCredentials:
    base_url: str
    session_token: str


@dataclass
class SolveRequest:
    prompt: str
    tripletex_credentials: TripletexCredentials
    files: list[FileAttachment] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "SolveRequest":
        creds = data["tripletex_credentials"]
        files = [
            FileAttachment(**f) for f in data.get("files", [])
        ]
        return cls(
            prompt=data["prompt"],
            tripletex_credentials=TripletexCredentials(**creds),
            files=files,
        )


@dataclass
class SolveResponse:
    status: str = "completed"

    def to_dict(self) -> dict:
        return {"status": self.status}


@dataclass
class AgentState:
    """State passed through the LangGraph agent."""
    messages: list = field(default_factory=list)
    credentials: TripletexCredentials | None = None
    files: list[FileAttachment] = field(default_factory=list)
