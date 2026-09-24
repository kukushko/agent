"""Core types for defining tool packages."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, ClassVar, Iterator, Mapping, Protocol, TypeVar, cast

from sandbox import Sandbox, UnavailableSandbox


ToolMethod = TypeVar("ToolMethod", bound=Callable[..., Any])
_TOOL_MARKER = "__agent_tool__"
_TOOL_METADATA = "__agent_tool_metadata__"
TOOL_SUCCESS_MARKER = "__agent_tool_success__"
TOOL_RESOURCE_USAGE_MARKER = "__agent_tool_resource_usage__"
_CURRENT_CONTEXT: ContextVar[ToolContext | None] = ContextVar(
    "tool_context", default=None
)


@dataclass(frozen=True)
class Delegation:
    """Declare whether and at what resource cost a tool may be delegated."""

    allowed: bool = True
    costs: Mapping[str, int] = field(default_factory=lambda: {"tool_calls": 1})
    quota_defaults: Mapping[str, int] = field(
        default_factory=lambda: {"tool_calls": 200}
    )
    instructions: tuple[str, ...] = ()
    reason: str | None = None


@dataclass(frozen=True)
class ToolExposure:
    """Declare which generic execution surfaces may expose a tool."""

    direct: bool = True
    delegated: bool = True
    orchestration_entrypoint: bool = False


@dataclass(frozen=True)
class ToolMetadata:
    """Portable behavior and prompt metadata owned by a tool declaration."""

    delegation: Delegation = field(default_factory=Delegation)
    exposure: ToolExposure = field(default_factory=ToolExposure)
    prompt_instructions: tuple[str, ...] = ()
    epistemic_roles: tuple[str, ...] = ()
    reliability_guidance: tuple[str, ...] = ()


def tool(
    method: ToolMethod | None = None,
    *,
    delegation: Delegation = Delegation(),
    exposure: ToolExposure = ToolExposure(),
    prompt_instructions: tuple[str, ...] = (),
    epistemic_roles: tuple[str, ...] = (),
    reliability_guidance: tuple[str, ...] = (),
) -> ToolMethod | Callable[[ToolMethod], ToolMethod]:
    """Mark a package method as a tool and attach portable metadata."""
    def decorate(candidate: ToolMethod) -> ToolMethod:
        setattr(candidate, _TOOL_MARKER, True)
        setattr(
            candidate,
            _TOOL_METADATA,
            ToolMetadata(
                delegation,
                exposure,
                tuple(prompt_instructions),
                tuple(epistemic_roles),
                tuple(reliability_guidance),
            ),
        )
        return candidate

    return decorate(method) if method is not None else decorate


def is_tool(method: Callable[..., Any]) -> bool:
    """Return whether a method was explicitly marked as a tool."""
    return bool(getattr(method, _TOOL_MARKER, False))


def tool_metadata(method: Callable[..., Any]) -> ToolMetadata:
    """Return metadata attached by the tool decorator."""
    return cast(ToolMetadata, getattr(method, _TOOL_METADATA, ToolMetadata()))


def with_tool_success(
    result: dict[str, object], success: bool
) -> dict[str, object]:
    """Attach internal success metadata consumed by the tool registry."""
    return {TOOL_SUCCESS_MARKER: success, **result}


def with_tool_resource_usage(
    result: dict[str, object], usage: Mapping[str, int]
) -> dict[str, object]:
    """Attach generic measured resource usage consumed by orchestration."""
    return {TOOL_RESOURCE_USAGE_MARKER: dict(usage), **result}


@dataclass(frozen=True)
class ToolContext:
    """Request-scoped state shared by all tools in one user turn."""

    user_input: str
    history: tuple[Any, ...]
    state: dict[str, Any] = field(default_factory=dict)


class ToolInvoker(Protocol):
    """Narrow interface for invoking another registered tool."""

    def execute(self, name: str, arguments: dict[str, Any]) -> Any:
        """Execute a tool through the shared runtime."""


ServiceType = TypeVar("ServiceType")


class ServiceRegistry:
    """Type-keyed long-lived services available to tool packages."""

    def __init__(self) -> None:
        self._values: dict[type[Any], Any] = {}

    def register(self, contract: type[ServiceType], service: ServiceType) -> None:
        if contract in self._values:
            raise RuntimeError(f"service is already registered: {contract.__name__}")
        self._values[contract] = service

    def optional(self, contract: type[ServiceType]) -> ServiceType | None:
        return cast(ServiceType | None, self._values.get(contract))

    def require(self, contract: type[ServiceType]) -> ServiceType:
        service = self.optional(contract)
        if service is None:
            raise RuntimeError(f"required service is unavailable: {contract.__name__}")
        return service


@dataclass
class ToolEnvironment:
    """Long-lived configuration and services available to tool packages."""

    work_dir: Path
    sandbox: Sandbox = field(default_factory=UnavailableSandbox)
    tool_parameters: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    services: ServiceRegistry = field(default_factory=ServiceRegistry)
    _tools: ToolInvoker | None = field(default=None, init=False, repr=False)

    @property
    def context(self) -> ToolContext:
        """Return the active request context or fail outside a user turn."""
        context = _CURRENT_CONTEXT.get()
        if context is None:
            raise RuntimeError(
                "ToolContext is not available outside an active user turn"
            )
        return context

    @property
    def tools(self) -> ToolInvoker:
        """Return the shared tool invoker after registry initialization."""
        if self._tools is None:
            raise RuntimeError("tool registry has not been initialized")
        return self._tools

    def bind_tools(self, tools: ToolInvoker) -> None:
        """Bind the registry once during application initialization."""
        if self._tools is not None:
            raise RuntimeError("tool registry is already initialized")
        self._tools = tools


@contextmanager
def activate_tool_context(context: ToolContext) -> Iterator[None]:
    """Activate a context for the duration of one user turn."""
    token: Token[ToolContext | None] = _CURRENT_CONTEXT.set(context)
    try:
        yield
    finally:
        _CURRENT_CONTEXT.reset(token)


class ToolPackage:
    """Base class for explicitly registered groups of related tools."""

    namespace: ClassVar[str]
    parameter_defaults: ClassVar[Mapping[str, str]] = {}

    def __init__(self, environment: ToolEnvironment) -> None:
        self.environment = environment

    def parameter(self, name: str) -> str:
        """Return a package parameter override or its package-owned default."""
        if name not in self.parameter_defaults:
            raise KeyError(f"unknown parameter for tool package {self.namespace}: {name}")
        return self.environment.tool_parameters.get(self.namespace, {}).get(
            name, self.parameter_defaults[name]
        )

    @classmethod
    def accepts_parameter(cls, name: str) -> bool:
        """Return whether a package owns a named configuration parameter."""
        return name in cls.parameter_defaults

    @classmethod
    def check_availability(cls, environment: ToolEnvironment) -> Availability:
        """Return whether this package can be registered in the environment."""
        return Availability(True)


@dataclass(frozen=True)
class Availability:
    """Availability result returned before a tool package is instantiated."""

    available: bool
    reason: str | None = None
