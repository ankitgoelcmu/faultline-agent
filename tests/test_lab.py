"""Run with: python -m pytest -q   (or: python tests/test_lab.py)"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lab.agents import ScriptedAgent
from lab.harness import Harness, CONDITIONS
from lab.rescore import replay
from lab.stats import holm, mcnemar_exact, paired_diff_ci, pass_hat_k, wilson
from lab.tasks import build_tasks, grade, mentions_status
from lab.world import World


def test_thirty_tasks():
    assert len(build_tasks()) == 30


def test_plans_pass_with_no_faults():
    for t in build_tasks():
        for c in CONDITIONS:
            w = World(t["data"])
            ScriptedAgent().run(t, Harness(w, c, 0.0, 1, []))
            ok, fails = grade(t, w)
            assert ok, (t["id"], c, fails)


def test_no_intent_key_collisions_in_reference_plans():
    # Model-free check: the idempotency key must never merge two distinct
    # intended writes of a task (e.g. two emails to one customer, or two
    # refunds of equal amount on one order). Collisions would make safe_retry
    # drop correct work; none exist in this task set.
    from lab.harness import intent_key
    from lab.world import WRITE_TOOLS
    for t in build_tasks():
        keys = [intent_key(tool, a) for tool, a in t["plan"] if tool in WRITE_TOOLS]
        assert len(keys) == len(set(keys)), t["id"]


def test_too_broad_key_merges_two_intended_writes():
    # Negative case for the key semantics: the email key is the recipient
    # alone, so two *different* intended emails to one customer collapse into
    # one under safe_retry, with no fault at all. Without keys both are sent.
    t = build_tasks()[5]
    cid = next(a["customer_id"] for tool, a in t["plan"] if tool == "send_email")
    first = {"customer_id": cid, "subject": "Your refund", "body": "Your refund was issued."}
    second = {"customer_id": cid, "subject": "Survey", "body": "How did we do?"}
    w = World(t["data"])
    h = Harness(w, "safe_retry", 0.0, 1, [])
    h.call("send_email", first)
    h.call("send_email", second)
    assert len(w.emails) == 1                 # the second, different email was dropped
    w2 = World(t["data"])
    h2 = Harness(w2, "retry", 0.0, 1, [])
    h2.call("send_email", first)
    h2.call("send_email", second)
    assert len(w2.emails) == 2


def test_doing_nothing_fails():
    for t in build_tasks():
        ok, _ = grade(t, World(t["data"]))
        assert not ok, t["id"]


def test_lost_response_duplicates_without_idempotency():
    t = build_tasks()[5]                      # a single refund task
    refund = next(a for tool, a in t["plan"] if tool == "issue_refund")
    w = World(t["data"])
    h = Harness(w, "retry", 1.0, 3, [])       # every call faults
    h.rng.random = iter([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).__next__  # force lost responses
    h.call("issue_refund", refund)
    assert len(w.refunds) > 1                 # naive retries refund more than once
    w2 = World(t["data"])
    h2 = Harness(w2, "safe_retry", 1.0, 3, [])
    h2.rng.random = iter([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).__next__
    h2.call("issue_refund", refund)
    assert len(w2.refunds) == 1               # idempotency key stops the duplicate


def test_status_emails_graded_by_meaning():
    yes = [("processing", "Your order is currently being processed."),
           ("processing", "It hasn't shipped yet."),
           ("processing", "Order O-0103 is in “processing” status."),
           ("shipped", "Good news: your order has shipped."),
           ("shipped", "It's on its way and should arrive Friday."),
           ("delivered", "Our records show it was delivered on 2026-09-30.")]
    no = [("shipped", "Your order hasn’t shipped yet."),
          ("shipped", "It has not yet been shipped."),
          ("shipped", "It will be shipped tomorrow."),
          ("delivered", "It has shipped and will be delivered by Friday."),
          ("delivered", "It was not delivered."),
          ("processing", "Your order has shipped.")]
    for status, text in yes:
        assert mentions_status(status, text), (status, text)
    for status, text in no:
        assert not mentions_status(status, text), (status, text)


def test_replay_rebuilds_the_final_state():
    for t in build_tasks():
        for c in CONDITIONS:
            trace = []
            w = World(t["data"])
            ScriptedAgent().run(t, Harness(w, c, 0.3, 7, trace))
            w2 = replay(t, trace, keyed=c in ("safe_retry", "guarded"))
            assert grade(t, w) == grade(t, w2), (t["id"], c)
            assert (w.refunds, w.emails, w.escalations) == (w2.refunds, w2.emails, w2.escalations)


def test_bursty_faults_keep_the_rate_but_cluster():
    w = World(build_tasks()[0]["data"])
    for model in ("uniform", "bursty"):
        h = Harness(w, "baseline", 0.10, 11, [], fault_model=model, burst_len=4.0)
        seq = [h._faulted() for _ in range(200_000)]
        rate = sum(seq) / len(seq)
        after_fault = sum(b for a, b in zip(seq, seq[1:]) if a) / sum(seq[:-1])
        assert 0.09 < rate < 0.11, (model, rate)
        if model == "uniform":
            assert after_fault < 0.12, after_fault     # independent: no memory
        else:
            assert after_fault > 0.5, after_fault      # an outage: the next call likely fails too
    # Short episodes must see the nominal rate too, not fewer faults.
    first14 = [f for s in range(20_000)
               for f in (lambda h: [h._faulted() for _ in range(14)])(
                   Harness(w, "baseline", 0.10, s, [], fault_model="bursty"))]
    assert 0.09 < sum(first14) / len(first14) < 0.11


def test_stale_read_hides_the_latest_write():
    t = next(t for t in build_tasks() if t["kind"] == "address_change")
    oid, addr = next((a["order_id"], a["address"]) for tool, a in t["plan"]
                     if tool == "update_shipping_address")
    old = t["data"]["orders"][oid]["ship_address"]
    trace = []
    h = Harness(World(t["data"]), "baseline", 0.0, 1, trace, stale_rate=1.0)
    assert h.call("get_order", {"order_id": oid})[1]["ship_address"] == old   # no write yet
    h.call("update_shipping_address", {"order_id": oid, "address": addr})
    ok, out = h.call("get_order", {"order_id": oid})
    assert ok and out["ship_address"] == old and trace[-1]["fault"] == "stale_read"
    assert h.world.orders[oid]["ship_address"] == addr    # the write did happen


def test_key_available_dedups_only_with_a_reused_key():
    from lab.world import schemas_with_keys, WRITE_TOOLS
    keyed = {t["name"] for t in schemas_with_keys() if "idempotency_key" in t["input_schema"]["properties"]}
    assert keyed == WRITE_TOOLS
    t = build_tasks()[5]
    refund = next(a for tool, a in t["plan"] if tool == "issue_refund")
    for key, expect in (("r-1", 1), (None, 2)):
        w = World(t["data"])
        h = Harness(w, "key_available", 1.0, 3, [], lost_share=1.0)
        h.rng.random = iter([0.0] * 10).__next__      # first attempt: lost response
        args = dict(refund, **({"idempotency_key": key} if key else {}))
        h.call("issue_refund", args)                   # applied, but the agent sees a timeout
        h.fault_rate = 0.0
        h.call("issue_refund", dict(args))             # the agent retries
        assert len(w.refunds) == expect, (key, w.refunds)
        assert replay(t, h.trace, keyed=False).refunds == w.refunds   # rescore matches


def test_wilson_interval():
    lo, hi = wilson(0, 12)
    assert lo == 0 and 0.24 < hi < 0.25      # 0/12 still allows up to ~24%
    lo, hi = wilson(150, 150)
    assert hi == 1 and lo > 0.97


def test_paired_stats():
    assert mcnemar_exact(0, 0) == 1.0
    assert abs(mcnemar_exact(0, 10) - 2 / 1024) < 1e-12     # 10 discordant pairs, all one way
    assert abs(mcnemar_exact(3, 3) - 1.0) < 1e-12
    assert holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]
    assert pass_hat_k(5, 5, 3) == 1 and pass_hat_k(4, 5, 5) == 0
    assert abs(pass_hat_k(4, 5, 2) - 6 / 10) < 1e-12      # C(4,2)/C(5,2)
    d, lo, hi = paired_diff_ci(119, 0, 28, 3)              # 28 discordant pairs, all one way
    assert abs(d - 28 / 150) < 1e-12 and 0 < lo < d < hi < 1
    d, lo, hi = paired_diff_ci(100, 12, 13, 25)            # balanced: interval spans 0
    assert lo < 0 < hi


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
