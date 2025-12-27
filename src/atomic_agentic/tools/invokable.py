from .base import Tool, ArgumentMap

# ───────────────────────────────────────────────────────────────────────────────
# Agent Tool
# ───────────────────────────────────────────────────────────────────────────────
class AgentTool(Tool):
    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def __init__(self, agent: Agent):
        # extract tool creation inputs
        function = agent.invoke
        name = "invoke"
        namespace = agent.name
        description = agent.description
        # set private variable
        self._agent = agent
        super().__init__(function, name, namespace, description)

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def agent(self) -> Agent:
        return self._agent
    
    @agent.setter
    def agent(self, value: Agent)-> None:
        self._agent = value
        self._function = self._agent.invoke
        self._namespace = value.name
        self._description = value.description
        # Identity in import space (may be overridden by subclasses)
        self._module, self._qualname = self._get_mod_qual(self.function)
        # Build argument schema and return type from the current function.
        self._arguments_map, self._return_type = self._build_io_schemas()
        # Persistibility flag exposed as a public property.
        self._is_persistible_internal: bool = self._compute_is_persistible()
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def namespace(self) -> str:
        return self._namespace
    
    @property
    def description(self) -> str:
        return self._description
    
    @property
    def function(self) -> Callable:
        return self._function
    
    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _build_io_schemas(self) -> tuple[ArgumentMap, str]:
        """Construct ``arguments_map`` and ``return_type`` from the wrapped
        callable's signature.

        Rules:
        - If an annotation is present, it *always* defines the type string.
        - If no annotation but a default value exists, the type string is
          derived from ``type(default)``.
        - If neither is present, the type string is 'Any'.
        """
        return self.agent.pre_invoke.arguments_map, self.agent.post_invoke.return_type

    def _compute_is_persistible(self) -> bool:
        """Default persistibility check for callable-based tools.

        A Tool is considered persistible if its function has both ``__module__``
        and ``__qualname__`` and does not appear to be a local/helper function.
        Subclasses can override this with their own criteria.
        """
        return self.agent.pre_invoke.is_persistible and self.agent.post_invoke.is_persistible

    def to_arg_kwarg(self, inputs: Mapping[str, Any]) -> tuple[tuple[Any, ...], Dict[str, Any]]:
        """Default implementation for mapping input dicts to ``(*args, **kwargs)``.

        The base policy is:

        - Required parameters (those without ``default`` and not VAR_*) must be present.
        - Unknown keys raise if there is no VAR_KEYWORD parameter; otherwise they
          are accepted and passed through in ``**kwargs``.
        - POSITIONAL_ONLY parameters are always passed positionally.
        - POSITIONAL_OR_KEYWORD and KEYWORD_ONLY parameters are passed as
          keywords (Python accepts this for both kinds).
        - VAR_POSITIONAL expects the mapping to contain the parameter name with
          a sequence value; these are appended to ``*args``.
        - VAR_KEYWORD collects all remaining unknown keys into ``**kwargs``.
        """
        return tuple([]), dict(inputs)

    def execute(self, args: tuple[Any, ...], kwargs: Dict[str, Any]) -> Any:
        """Execute the underlying callable.

        Subclasses may override this to change *how* a tool is executed (for
        example, by making a remote MCP call or invoking an Agent), but should
        not change the high-level semantics.
        """
        try:
            result = self._function(kwargs) # function = self.agent.invoke()
        except Exception as e:  # pragma: no cover - thin wrapper
            raise ToolInvocationError(f"{self.full_name}: invocation failed: {e}") from e
        return result

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self)-> OrderedDict[str, Any]:
        base = super().to_dict()
        base.update(OrderedDict(
            agent = self.agent.to_dict()
        ))
        return base
