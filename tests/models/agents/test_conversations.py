from __future__ import annotations

import threading
from types import MappingProxyType

import pytest

from atomic_agentic.models.agents.context import ContextVariable
from atomic_agentic.models.agents.conversations import Conversation
from atomic_agentic.models.agents.records import AgentRecord


def make_conv(**variables: ContextVariable) -> Conversation:
    return Conversation(context=dict(variables))


def cv(name: str, value: object = 1, **kwargs: object) -> ContextVariable:
    return ContextVariable(name=name, value=value, **kwargs)  # type: ignore[arg-type]


class TestConversationContextConstruction:
    def test_defaults_to_empty_read_only_view(self) -> None:
        conv = Conversation()

        assert isinstance(conv.context, MappingProxyType)
        assert len(conv.context) == 0
        with pytest.raises(TypeError):
            conv.context["x"] = cv("x")  # type: ignore[index]

    def test_constructor_copies_and_shares_variables_by_identity(self) -> None:
        var = cv("x", [1])
        source = {"x": var}

        conv = Conversation(context=source)
        source["y"] = cv("y")

        assert "y" not in conv.context
        assert conv.context["x"] is var
        assert conv.context["x"].value is var.value

    def test_rejects_bad_context(self) -> None:
        with pytest.raises(ValueError, match="Mapping"):
            Conversation(context=[])  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="ContextVariable"):
            Conversation(context={"x": 1})  # type: ignore[dict-item]
        with pytest.raises(ValueError, match="must equal"):
            Conversation(context={"y": cv("x")})

    def test_rejects_reserved_task_result_name(self) -> None:
        with pytest.raises(ValueError, match="reserved"):
            Conversation(context={"task_result_0": cv("task_result_0")})

    def test_slice_keeps_working_with_clean_context(self) -> None:
        conv = make_conv(x=cv("x"))

        assert conv.slice().context["x"] is conv.context["x"]

    def test_equality_is_by_value_over_every_field(self) -> None:
        a = Conversation(context={"x": cv("x")})
        same = Conversation(context={"x": cv("x")})

        assert a == same
        assert a != Conversation()
        assert a != Conversation(context={"x": cv("x", 2)})

    def test_unhashable(self) -> None:
        with pytest.raises(TypeError):
            hash(Conversation())


class TestRemember:
    def test_new_key_defaults_not_readonly(self) -> None:
        conv = Conversation()

        conv.remember("x", 5, "five")

        assert conv.context["x"] == cv("x", 5, description="five")
        assert conv.context["x"].readonly is False

    def test_readonly_creation_and_key_normalization(self) -> None:
        conv = Conversation()

        conv.remember(" x ", 1, readonly=True)

        assert conv.context["x"].readonly is True
        with pytest.raises(ValueError):
            conv.remember("1bad", 1)

    @pytest.mark.parametrize("key", ["task_result_0", " task_result_x "])
    def test_reserved_task_result_name_raises_and_leaves_context(self, key: str) -> None:
        conv = make_conv(x=cv("x"))
        before = conv.context

        with pytest.raises(ValueError, match="reserved"):
            conv.remember(key, 1)

        assert conv.context is before

    @pytest.mark.parametrize("readonly", [False, True])
    def test_existing_name_raises_and_leaves_context(self, readonly: bool) -> None:
        conv = make_conv(x=cv("x", 1, readonly=readonly))
        before = conv.context

        with pytest.raises(ValueError, match="already exists"):
            conv.remember("x", 2)

        assert conv.context is before
        assert conv.context["x"].value == 1

    def test_earlier_view_is_unchanged(self) -> None:
        conv = make_conv(x=cv("x", 1))
        before = conv.context

        conv.remember("y", 2)

        assert "y" not in before
        assert "y" in conv.context


class TestDescribe:
    def test_replaces_description_only(self) -> None:
        var = cv("x", [1], description="old")
        conv = make_conv(x=var)

        conv.describe("x", "new")

        assert conv.context["x"].description == "new"
        assert conv.context["x"].value is var.value

    def test_none_and_blank_clear(self) -> None:
        conv = make_conv(x=cv("x", description="old"))

        conv.describe("x", "   ")

        assert conv.context["x"].description is None

    def test_non_string_raises(self) -> None:
        conv = make_conv(x=cv("x"))

        with pytest.raises(TypeError):
            conv.describe("x", 3)  # type: ignore[arg-type]

    def test_unknown_key_raises_and_readonly_is_skipped(self) -> None:
        conv = make_conv(x=cv("x", readonly=True))
        before = conv.context

        with pytest.raises(KeyError):
            conv.describe("nope", "d")
        assert conv.describe("x", "d") is False
        assert conv.context is before

    def test_returns_true_when_written(self) -> None:
        conv = make_conv(x=cv("x"))

        assert conv.describe("x", "d") is True
        assert conv.context["x"].description == "d"


class TestUpdate:
    def test_replaces_value_only(self) -> None:
        conv = make_conv(x=cv("x", 1, description="d", render_mode="r", preview_limit=3))
        before = conv.context

        assert conv.update("x", 2) is True

        stored = conv.context["x"]
        assert (stored.value, stored.description) == (2, "d")
        assert (stored.render_mode, stored.preview_limit) == ("r", 3)
        assert before["x"].value == 1

    def test_readonly_is_skipped(self) -> None:
        conv = make_conv(x=cv("x", 1, readonly=True))
        before = conv.context

        assert conv.update("x", 2) is False
        assert conv.context is before

    def test_unknown_key_raises(self) -> None:
        with pytest.raises(KeyError):
            Conversation().update("nope", 1)


class TestForget:
    def test_removes_key_and_prior_view_unchanged(self) -> None:
        conv = make_conv(x=cv("x"), y=cv("y"))
        before = conv.context

        conv.forget("x")

        assert "x" not in conv.context
        assert "x" in before

    def test_unknown_key_raises(self) -> None:
        with pytest.raises(KeyError):
            Conversation().forget("x")

    def test_readonly_can_be_forgotten(self) -> None:
        conv = make_conv(x=cv("x", readonly=True))

        conv.forget("x")

        assert "x" not in conv.context


class TestBulk:
    def test_remember_many_reserved_name_is_atomic(self) -> None:
        conv = make_conv(x=cv("x"))
        before = conv.context

        with pytest.raises(ValueError, match="reserved"):
            conv.remember_many([cv("ok"), cv("task_result_1")])

        assert conv.context is before

    def test_remember_many_adds_new_names(self) -> None:
        conv = make_conv(x=cv("x", 1))

        conv.remember_many([cv("y", 2), cv("z", 3)])

        assert conv.context["x"].value == 1
        assert (conv.context["y"].value, conv.context["z"].value) == (2, 3)

    @pytest.mark.parametrize(
        "items, exc",
        [
            ([cv("a"), cv("a")], ValueError),
            ([cv("a"), cv("r")], ValueError),
            ([cv("a"), cv("a2"), cv("r")], ValueError),
            ([cv("a"), 1], TypeError),
        ],
    )
    def test_remember_many_failure_leaves_same_view(self, items: list, exc: type) -> None:
        conv = make_conv(r=cv("r", readonly=True))
        before = conv.context

        with pytest.raises(exc):
            conv.remember_many(items)

        assert conv.context is before

    def test_forget_many_removes_all_including_readonly_and_dupes(self) -> None:
        conv = make_conv(x=cv("x"), y=cv("y", readonly=True), z=cv("z"))

        conv.forget_many(["x", "y", "x"])

        assert list(conv.context) == ["z"]

    def test_forget_many_unknown_removes_nothing(self) -> None:
        conv = make_conv(x=cv("x"))
        before = conv.context

        with pytest.raises(KeyError):
            conv.forget_many(["x", "nope"])

        assert conv.context is before


class TestPreviewLimits:
    LIMITS = ("response_preview_limit", "prompt_preview_limit", "context_preview_limit")

    def test_default_to_none(self) -> None:
        conv = Conversation()

        assert all(getattr(conv, name) is None for name in self.LIMITS)

    @pytest.mark.parametrize("value", [None, 0, 1, 10_000])
    def test_valid_values_accepted_at_construction_and_assignment(
        self, value: int | None
    ) -> None:
        for name in self.LIMITS:
            conv = Conversation(**{name: value})
            assert getattr(conv, name) == value

            other = Conversation()
            setattr(other, name, value)
            assert getattr(other, name) == value

    @pytest.mark.parametrize("value", [-1, True, False, 1.5, "5"])
    def test_invalid_values_rejected_at_construction_and_assignment(
        self, value: object
    ) -> None:
        for name in self.LIMITS:
            with pytest.raises(ValueError, match=name):
                Conversation(**{name: value})  # type: ignore[arg-type]

            conv = Conversation(**{name: 7})  # type: ignore[arg-type]
            with pytest.raises(ValueError, match=name):
                setattr(conv, name, value)
            assert getattr(conv, name) == 7

    def test_limits_take_part_in_equality(self) -> None:
        a = Conversation(response_preview_limit=5)
        b = Conversation(response_preview_limit=5)
        c = Conversation(response_preview_limit=6)

        assert a == b
        assert a != c

    def test_context_mutators_still_work_after_limit_assignment(self) -> None:
        conv = Conversation()
        conv.context_preview_limit = 3
        conv.remember("k", 1)

        assert conv.context["k"].value == 1
        assert isinstance(conv.context, MappingProxyType)


class TestHistoryRenderMode:
    def test_defaults_to_str(self) -> None:
        assert Conversation().render_mode == "s"

    @pytest.mark.parametrize("mode", ["s", "r", "a", "j", "p"])
    def test_valid_modes_accepted_at_construction_and_assignment(self, mode: str) -> None:
        assert Conversation(render_mode=mode).render_mode == mode

        conv = Conversation()
        conv.render_mode = mode
        assert conv.render_mode == mode

    @pytest.mark.parametrize("mode", ["x", "S", "", None, 1])
    def test_invalid_modes_rejected_and_old_value_kept(self, mode: object) -> None:
        with pytest.raises(ValueError, match="render_mode"):
            Conversation(render_mode=mode)  # type: ignore[arg-type]

        conv = Conversation(render_mode="j")
        with pytest.raises(ValueError, match="render_mode"):
            conv.render_mode = mode  # type: ignore[assignment]
        assert conv.render_mode == "j"

    def test_takes_part_in_equality(self) -> None:
        a = Conversation(render_mode="j")

        assert a == Conversation(render_mode="j")
        assert a != Conversation()


class TestRememberRenderMode:
    def test_remember_stores_render_mode(self) -> None:
        conv = Conversation()
        conv.remember("k", {"a": 1}, render_mode="j")

        assert conv.context["k"].render_mode == "j"

    def test_remember_defaults_to_str_mode(self) -> None:
        conv = Conversation()
        conv.remember("k", 1)

        assert conv.context["k"].render_mode == "s"

    def test_update_keeps_render_mode(self) -> None:
        conv = Conversation()
        conv.remember("k", 1, render_mode="r")
        conv.update("k", 2)

        assert conv.context["k"].render_mode == "r"

    def test_remember_rejects_none_render_mode(self) -> None:
        conv = Conversation()

        with pytest.raises(ValueError, match="render_mode"):
            conv.remember("k", 1, render_mode=None)  # type: ignore[arg-type]

    def test_remember_rejects_invalid_render_mode(self) -> None:
        conv = Conversation()

        with pytest.raises(ValueError, match="render_mode"):
            conv.remember("k", 1, render_mode="bogus")
        assert "k" not in conv.context

    def test_remember_stores_and_update_keeps_preview_limit(self) -> None:
        conv = Conversation()
        conv.remember("k", 1, preview_limit=4)
        assert conv.context["k"].preview_limit == 4

        conv.update("k", 2)
        assert conv.context["k"].preview_limit == 4

    def test_remember_rejects_invalid_preview_limit(self) -> None:
        conv = Conversation()

        with pytest.raises(ValueError, match="preview_limit"):
            conv.remember("k", 1, preview_limit=-1)
        assert "k" not in conv.context

    def test_describe_keeps_render_mode(self) -> None:
        conv = Conversation()
        conv.remember("k", 1, "old", render_mode="a")
        conv.describe("k", "new")

        assert conv.context["k"].render_mode == "a"
        assert conv.context["k"].description == "new"


def rec(prompt: str = "p", **kwargs: object) -> AgentRecord:
    return AgentRecord(user_prompt=prompt, generated_response="r", **kwargs)  # type: ignore[arg-type]


class TestNoName:
    def test_name_is_not_a_field(self) -> None:
        with pytest.raises(TypeError):
            Conversation(name="c")  # type: ignore[call-arg]
        assert not hasattr(Conversation(), "name")

    def test_lock_is_not_an_init_arg_repr_or_equality_part(self) -> None:
        with pytest.raises(TypeError):
            Conversation(_lock=threading.RLock())  # type: ignore[call-arg]
        conv = Conversation()

        assert "_lock" not in repr(conv)
        assert conv == Conversation()
        assert conv._lock is not Conversation()._lock


class TestSlice:
    def make(self, n: int = 5) -> Conversation:
        return Conversation(records=[rec(str(i)) for i in range(n)])

    def prompts(self, conv: Conversation) -> list[str]:
        return [r.user_prompt for r in conv.records]

    def test_range_is_end_exclusive(self) -> None:
        assert self.prompts(self.make().slice(1, 3)) == ["1", "2"]

    def test_defaults_copy_everything(self) -> None:
        conv = self.make()
        copy = conv.slice()

        assert copy.records == conv.records
        assert copy.records is not conv.records

    def test_negative_indices(self) -> None:
        conv = self.make()

        assert self.prompts(conv.slice(-2)) == ["3", "4"]
        assert self.prompts(conv.slice(None, -3)) == ["0", "1"]

    def test_out_of_range_is_clamped_and_empty_is_legal(self) -> None:
        conv = self.make(3)

        assert self.prompts(conv.slice(1, 99)) == ["1", "2"]
        assert conv.slice(10, 20).records == []
        assert conv.slice(2, 1).records == []

    def test_records_are_reference_copied(self) -> None:
        conv = self.make()

        assert conv.slice(0, 2).records[0] is conv.records[0]

    def test_independent_of_source(self) -> None:
        conv = self.make(2)
        copy = conv.slice()

        copy.records.append(rec("new"))
        copy.remember("k", 1)
        copy.response_preview_limit = 9

        assert len(conv.records) == 2
        assert "k" not in conv.context
        assert conv.response_preview_limit is None

    def test_carries_latest_context_in_distinct_container(self) -> None:
        var = cv("x", [1])
        conv = Conversation(records=[rec()], context={"x": var})
        copy = conv.slice(0, 0)

        assert copy.context is not conv.context
        assert copy.context["x"] is var

    def test_copies_limits_and_render_mode_and_gets_fresh_lock(self) -> None:
        conv = Conversation(
            response_preview_limit=1,
            prompt_preview_limit=2,
            context_preview_limit=3,
            render_mode="j",
        )
        copy = conv.slice()

        assert (copy.response_preview_limit, copy.prompt_preview_limit, copy.context_preview_limit) == (1, 2, 3)
        assert copy.render_mode == "j"
        assert copy._lock is not conv._lock


class TestCommit:
    def test_appends_record(self) -> None:
        conv = Conversation()
        r = rec()

        conv.commit(r, {})

        assert conv.records == [r]

    def test_links_prev_children_only_when_prev_set(self) -> None:
        conv = Conversation()
        first = rec("1")
        second = rec("2", prev=first)

        conv.commit(first, {})
        conv.commit(second, {})

        assert first.children == [second]
        assert second.children == []

    def test_write_back_updates_eligible_keys(self) -> None:
        conv = make_conv(x=cv("x", 1, description="d"))

        conv.commit(rec(), {"x": cv("x", 2, description="d")})

        assert conv.context["x"].value == 2

    def test_absent_name_is_created_as_given(self) -> None:
        conv = make_conv(keep=cv("keep", 1))
        new = cv("new", 7, description="d", render_mode="r")

        conv.commit(rec(), {"new": new})

        assert conv.context["new"] is new
        assert conv.context["keep"].value == 1

    def test_readonly_is_skipped_and_held_writable_keeps_its_metadata(self) -> None:
        conv = make_conv(
            ro=cv("ro", 1, readonly=True), keep=cv("keep", 1, description="d", preview_limit=3)
        )

        conv.commit(rec(), {"ro": cv("ro", 99), "keep": cv("keep", 5, description="x")})

        assert conv.context["ro"].value == 1
        assert conv.context["keep"].value == 5
        assert conv.context["keep"].description == "d"
        assert conv.context["keep"].preview_limit == 3

    def test_only_passed_names_are_touched(self) -> None:
        conv = make_conv(a=cv("a", 1), b=cv("b", 1))

        conv.commit(rec(), {"a": cv("a", 2)})

        assert (conv.context["a"].value, conv.context["b"].value) == (2, 1)

    def test_latest_writer_wins_and_held_name_is_not_recreated(self) -> None:
        conv = Conversation()

        conv.commit(rec("1"), {"x": cv("x", 1, description="first")})
        conv.commit(rec("2"), {"x": cv("x", 2, description="second")})

        assert conv.context["x"].value == 2
        assert conv.context["x"].description == "first"

    def test_concurrent_commits_keep_every_record_and_child(self) -> None:
        conv = Conversation()
        root = rec("root")
        conv.commit(root, {})
        n_threads, per_thread = 8, 25

        def work(t: int) -> None:
            for i in range(per_thread):
                conv.commit(rec(f"{t}-{i}", prev=root), {})

        threads = [threading.Thread(target=work, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(conv.records) == 1 + n_threads * per_thread
        assert len(root.children) == n_threads * per_thread

    def test_concurrent_context_writes_all_land(self) -> None:
        conv = Conversation()

        def work(t: int) -> None:
            for i in range(25):
                conv.remember(f"k{t}_{i}", i)

        threads = [threading.Thread(target=work, args=(t,)) for t in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(conv.context) == 8 * 25

    def test_concurrent_updates_to_distinct_keys_all_land(self) -> None:
        conv = make_conv(**{f"k{t}": cv(f"k{t}", 0) for t in range(8)})

        def work(t: int) -> None:
            for i in range(1, 26):
                conv.update(f"k{t}", i)

        threads = [threading.Thread(target=work, args=(t,)) for t in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(conv.context[f"k{t}"].value == 25 for t in range(8))
