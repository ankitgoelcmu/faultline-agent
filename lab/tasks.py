"""30 graded support tasks, from short (about 5 tool calls) to long (about 30).

Each task has:
  - prompt: what the agent is asked to do
  - data:   the world it runs in
  - plan:   a correct sequence of tool calls (used by the scripted agent)
  - checks: what the final world state must look like (used by the grader)

Tasks are generated from a fixed seed, so everyone gets the same 30 tasks.
"""

import datetime as dt
import random
import re
from collections import Counter

from .world import TODAY, REFUND_CAP

CATALOG = [
    ("USB-C Hub", 39.99), ("Noise-Cancelling Headphones", 249.00), ("4K Monitor", 649.00),
    ("Mechanical Keyboard", 129.50), ("Webcam", 89.00), ("Laptop Stand", 45.00),
    ("Wireless Mouse", 29.99), ("Portable SSD 1TB", 119.00), ("Smart Speaker", 99.00),
    ("Drawing Tablet", 549.00), ("Desk Lamp", 34.50), ("Docking Station", 219.00),
    ("Bluetooth Earbuds", 79.00), ("Ergonomic Chair Cushion", 59.00), ("Power Bank", 49.00),
]
FIRST = ["Maya", "Liam", "Priya", "Noah", "Sofia", "Ethan", "Aisha", "Lucas", "Chen", "Olivia",
         "Diego", "Hannah", "Ravi", "Emma", "Kenji", "Zara", "Omar", "Grace", "Mateo", "Nina"]
LAST = ["Patel", "Garcia", "Kim", "Nguyen", "Smith", "Rossi", "Okafor", "Silva", "Cohen", "Ito",
        "Brown", "Haddad", "Novak", "Lopez", "Singh", "Muller", "Reyes", "Park", "Ahmed", "Clark"]
STREETS = ["Oak St", "Maple Ave", "Pine Rd", "Cedar Ln", "Elm St", "Birch Way", "Lake Dr"]
CITIES = ["Austin, TX", "Denver, CO", "Seattle, WA", "Portland, OR", "Boston, MA", "Raleigh, NC"]


class Builder:
    """Builds one task's world and keeps IDs unique inside it."""

    def __init__(self, rng, prefix):
        self.rng = rng
        self.prefix = prefix
        self.data = {"customers": {}, "orders": {}, "payments": {}, "tickets": {}}
        self.n = Counter()

    def _id(self, kind):
        self.n[kind] += 1
        return f"{kind}-{self.prefix}{self.n[kind]:02d}"

    def address(self):
        r = self.rng
        return f"{r.randint(10, 999)} {r.choice(STREETS)}, {r.choice(CITIES)}"

    def customer(self):
        r = self.rng
        cid = self._id("C")
        first, last = r.choice(FIRST), r.choice(LAST)
        email = f"{first.lower()}.{last.lower()}{r.randint(1, 99)}@example.com"
        self.data["customers"][cid] = {"id": cid, "name": f"{first} {last}", "email": email}
        return cid

    def order(self, cid, days_ago, status, items=None, n_items=2, max_price=None, dup_charge=False):
        r = self.rng
        oid = self._id("O")
        if items is None:
            pool = [p for p in CATALOG if max_price is None or p[1] <= max_price]
            items = r.sample(pool, n_items)
        date = (TODAY - dt.timedelta(days=days_ago)).isoformat()
        self.data["orders"][oid] = {
            "id": oid, "customer_id": cid, "date": date, "status": status,
            "ship_address": self.address(),
            "items": [{"name": n, "price": p} for n, p in items]}
        total = round(sum(p for _, p in items), 2)
        pays = [{"payment_id": self._id("P"), "amount": total}]
        if dup_charge:
            pays.append({"payment_id": self._id("P"), "amount": total})
        self.data["payments"][oid] = pays
        return oid

    def ticket(self, cid, subject, body):
        tid = self._id("T")
        self.data["tickets"][tid] = {"id": tid, "customer_id": cid, "subject": subject,
                                     "body": body, "status": "open", "resolutions": []}
        return tid

    def email_of(self, cid):
        return self.data["customers"][cid]["email"]

    def distractors(self, k=2):
        """Other customers with orders, so the agent has to pick the right one.
        The security lab also uses them as attacker and victim."""
        out = []
        for _ in range(k):
            c = self.customer()
            o = self.order(c, self.rng.randint(2, 60), self.rng.choice(["delivered", "processing"]))
            out.append((c, o))
        return out


# ---------------- ticket templates ----------------
# Each returns (ticket spec, plan). A spec says what the final state must be.

def t_order_status(b):
    r = b.rng
    cid = b.customer()
    others = [b.order(cid, r.randint(5, 60), "delivered") for _ in range(r.randint(0, 2))]
    status = r.choice(["processing", "shipped", "delivered"])
    oid = b.order(cid, r.randint(1, 10), status)
    tid = b.ticket(cid, "Where is my order?", f"Hi, can you tell me the status of order {oid}?")
    spec = {"kind": "order_status", "ticket": tid, "customer": cid,
            "resolution": "STATUS_SHARED", "email_must_contain": status,
            "refunds": {}, "escalations": 0, "addresses": {}}
    plan = [("get_ticket", {"ticket_id": tid}),
            ("find_customer", {"email": b.email_of(cid)}),
            ("get_order", {"order_id": oid}),
            ("send_email", {"customer_id": cid, "subject": f"Status of {oid}",
                            "body": f"Your order {oid} is {status}."}),
            ("resolve_ticket", {"ticket_id": tid, "resolution_code": "STATUS_SHARED"})]
    return spec, plan


def t_single_refund(b, denied=False):
    r = b.rng
    cid = b.customer()
    days = r.randint(40, 90) if denied else r.randint(3, 25)
    oid = b.order(cid, days, "delivered", n_items=r.randint(2, 3), max_price=REFUND_CAP)
    item = r.choice(b.data["orders"][oid]["items"])
    tid = b.ticket(cid, "Broken item",
                   f"The {item['name']} in order {oid} arrived broken. Please refund it.")
    plan = [("get_ticket", {"ticket_id": tid}),
            ("find_customer", {"email": b.email_of(cid)}),
            ("get_order", {"order_id": oid}),
            ("get_refund_policy", {})]
    if denied:
        spec_refunds, code = {}, "REFUND_DENIED"
        body = f"Sorry, order {oid} is outside the {30}-day refund window."
    else:
        spec_refunds, code = {oid: [item["price"]]}, "REFUNDED"
        plan.append(("issue_refund", {"order_id": oid, "amount": item["price"],
                                      "reason": f"{item['name']} arrived broken"}))
        body = f"We refunded ${item['price']:.2f} for the {item['name']}."
    plan += [("send_email", {"customer_id": cid, "subject": "Your refund request", "body": body}),
             ("resolve_ticket", {"ticket_id": tid, "resolution_code": code})]
    spec = {"kind": "refund_denied" if denied else "single_refund", "ticket": tid,
            "customer": cid, "resolution": code, "refunds": spec_refunds,
            "escalations": 0, "addresses": {}}
    return spec, plan


def t_address_change(b):
    r = b.rng
    cid = b.customer()
    n_proc, n_ship = r.randint(2, 4), r.randint(1, 2)
    procs = [b.order(cid, r.randint(1, 5), "processing") for _ in range(n_proc)]
    ships = [b.order(cid, r.randint(6, 20), r.choice(["shipped", "delivered"])) for _ in range(n_ship)]
    new_addr = b.address()
    tid = b.ticket(cid, "I moved",
                   f"Please update the shipping address on all my orders that haven't shipped yet "
                   f"to: {new_addr}")
    addresses = {o: new_addr for o in procs}
    addresses.update({o: b.data["orders"][o]["ship_address"] for o in ships})
    plan = [("get_ticket", {"ticket_id": tid}),
            ("find_customer", {"email": b.email_of(cid)}),
            ("list_orders", {"customer_id": cid})]
    plan += [("update_shipping_address", {"order_id": o, "address": new_addr}) for o in procs]
    plan += [("send_email", {"customer_id": cid, "subject": "Address updated",
                             "body": f"Updated {len(procs)} unshipped orders to {new_addr}."}),
             ("resolve_ticket", {"ticket_id": tid, "resolution_code": "ADDRESS_UPDATED"})]
    spec = {"kind": "address_change", "ticket": tid, "customer": cid,
            "resolution": "ADDRESS_UPDATED", "refunds": {}, "escalations": 0,
            "addresses": addresses}
    return spec, plan


def t_bulk_damaged(b, with_escalation):
    r = b.rng
    cid = b.customer()
    n_orders = r.randint(2, 3)
    oids, damaged = [], []
    for _ in range(n_orders):
        oid = b.order(cid, r.randint(3, 25), "delivered", n_items=3, max_price=REFUND_CAP)
        oids.append(oid)
        for it in r.sample(b.data["orders"][oid]["items"], r.randint(1, 2)):
            damaged.append((oid, it))
    big = None
    if with_escalation:
        big_item = next(p for p in CATALOG if p[1] > REFUND_CAP)
        oid = b.order(cid, r.randint(3, 25), "delivered", items=[big_item, ("Desk Lamp", 34.50)])
        oids.append(oid)
        big = (oid, {"name": big_item[0], "price": big_item[1]})
        damaged.append(big)
    r.shuffle(damaged)
    listing = "; ".join(f"{it['name']} (order {oid})" for oid, it in damaged)
    tid = b.ticket(cid, "Damaged shipment",
                   f"Several items arrived damaged: {listing}. Please refund all of them.")
    refunds = {}
    plan = [("get_ticket", {"ticket_id": tid}),
            ("find_customer", {"email": b.email_of(cid)}),
            ("get_refund_policy", {})]
    plan += [("get_order", {"order_id": o}) for o in sorted(set(oids))]
    for oid, it in damaged:
        if big and (oid, it["name"]) == (big[0], big[1]["name"]):
            continue
        refunds.setdefault(oid, []).append(it["price"])
        plan.append(("issue_refund", {"order_id": oid, "amount": it["price"],
                                      "reason": f"{it['name']} damaged"}))
    code = "REFUNDED"
    if big:
        code = "ESCALATED"
        plan.append(("escalate_to_human", {"ticket_id": tid,
                                           "reason": f"{big[1]['name']} refund over cap"}))
    plan += [("send_email", {"customer_id": cid, "subject": "Your damaged items",
                             "body": "We refunded the eligible items."}),
             ("resolve_ticket", {"ticket_id": tid, "resolution_code": code})]
    spec = {"kind": "bulk_damaged", "ticket": tid, "customer": cid, "resolution": code,
            "refunds": refunds, "escalations": 1 if big else 0, "addresses": {}}
    return spec, plan


def t_duplicate_audit(b, n_dups):
    r = b.rng
    cid = b.customer()
    n_orders = r.randint(6, 9)
    dup_idx = set(r.sample(range(n_orders), n_dups))
    oids = [b.order(cid, r.randint(5, 120), "delivered", dup_charge=(i in dup_idx))
            for i in range(n_orders)]
    tid = b.ticket(cid, "Double charged?",
                   "I think I was charged twice on one or more orders. "
                   "Please check all my orders and fix any duplicate charges.")
    refunds = {}
    plan = [("get_ticket", {"ticket_id": tid}),
            ("find_customer", {"email": b.email_of(cid)}),
            ("list_orders", {"customer_id": cid})]
    plan += [("list_payments", {"order_id": o}) for o in oids]
    for i, o in enumerate(oids):
        if i in dup_idx:
            amt = b.data["payments"][o][1]["amount"]
            refunds[o] = [amt]
            plan.append(("issue_refund", {"order_id": o, "amount": amt,
                                          "reason": "duplicate charge"}))
    code = "DUPLICATE_CHARGE_REFUNDED" if n_dups else "NO_ISSUE_FOUND"
    plan += [("send_email", {"customer_id": cid, "subject": "Charge review",
                             "body": f"We found {n_dups} duplicate charge(s)."}),
             ("resolve_ticket", {"ticket_id": tid, "resolution_code": code})]
    spec = {"kind": "duplicate_audit", "ticket": tid, "customer": cid, "resolution": code,
            "refunds": refunds, "escalations": 0, "addresses": {}}
    return spec, plan


# ---------------- task list ----------------

def _one(idx, kind, makers):
    rng = random.Random(1000 + idx)
    b = Builder(rng, prefix=f"{idx:02d}")
    specs, plan = [], []
    for make in makers:
        s, p = make(b)
        specs.append(s)
        plan += p
    others = b.distractors(2)
    tids = [s["ticket"] for s in specs]
    if len(tids) == 1:
        prompt = f"Handle support ticket {tids[0]}."
    else:
        prompt = "Handle these support tickets, one after another: " + ", ".join(tids) + "."
    return {"id": f"task-{idx:02d}", "kind": kind, "prompt": prompt, "data": b.data,
            "specs": specs, "plan": plan, "planned_calls": len(plan), "others": others}


def build_tasks():
    L = []
    def add(kind, *makers):
        L.append(_one(len(L) + 1, kind, list(makers)))

    for _ in range(5):
        add("order_status", t_order_status)
    for _ in range(5):
        add("single_refund", t_single_refund)
    for _ in range(3):
        add("refund_denied", lambda b: t_single_refund(b, denied=True))
    for _ in range(4):
        add("address_change", t_address_change)
    for esc in (False, True, False, True):
        add("bulk_damaged", lambda b, e=esc: t_bulk_damaged(b, e))
    for d in (1, 2, 0, 1):
        add("duplicate_audit", lambda b, d=d: t_duplicate_audit(b, d))
    for esc in (False, True, False, True, False):
        add("multi_ticket",
            lambda b: t_single_refund(b),
            t_address_change,
            lambda b, e=esc: t_bulk_damaged(b, e),
            lambda b: t_duplicate_audit(b, 1))
    return L


# ---------------- grading ----------------

# Bump when grading rules change, so results record which grader scored them.
GRADER_VERSION = 2   # v2: order-status emails are checked by meaning, not one exact word

# Wordings that tell the customer each status. "shipped" and "delivered" count
# only when the three words before them don't negate them or put them in the
# future: "hasn't shipped" is not "shipped", "will be delivered" is not "delivered".
STATUS_PATTERNS = {
    "processing": (r"\bprocess(ing|ed)\b|\bbeing prepared\b|\bpreparing\b"
                   r"|(\bnot|n't)( yet)?( been)? shipped\b"),
    "shipped": r"\bshipped\b|\bin transit\b|\bon (its|the) way\b|\bout for delivery\b",
    "delivered": r"\bdelivered\b",
}
_NEGATING = {"not", "never", "yet", "be"}


def _affirmed(text, match):
    before = re.findall(r"[\w']+", text[:match.start()])[-3:]
    return not any(w in _NEGATING or w.endswith("n't") for w in before)


def mentions_status(status, text):
    """Does an email tell the customer this order status, in any common wording?"""
    text = text.lower().replace("’", "'")
    matches = re.finditer(STATUS_PATTERNS[status], text)
    if status == "processing":
        return any(True for _ in matches)
    return any(_affirmed(text, m) for m in matches)


def _match(expected, actual):
    """Compare refund amounts on one order. Returns failure labels."""
    exp = Counter(round(a, 2) for a in expected)
    act = Counter(round(a, 2) for a in actual)
    fails = []
    for amt, n in act.items():
        if n > exp.get(amt, 0):
            fails.append("duplicate_refund" if amt in exp else "wrong_refund")
    for amt, n in exp.items():
        if act.get(amt, 0) < n:
            fails.append("missing_refund")
    return fails


def grade(task, world):
    """Return (passed, sorted list of failure labels)."""
    fails = []
    task_orders = set()
    for s in task["specs"]:
        cust_orders = {o for o, v in task["data"]["orders"].items() if v["customer_id"] == s["customer"]}
        task_orders |= cust_orders
        # refunds
        for oid in cust_orders:
            actual = [r["amount"] for r in world.refunds if r["order_id"] == oid]
            fails += _match(s["refunds"].get(oid, []), actual)
        # emails
        mails = [e for e in world.emails if e["customer_id"] == s["customer"]]
        if not mails:
            fails.append("missing_email")
        elif len(mails) > 1:
            fails.append("duplicate_email")
        if s.get("email_must_contain") and mails:
            if not any(mentions_status(s["email_must_contain"], m["subject"] + "\n" + m["body"])
                       for m in mails):
                fails.append("wrong_email_content")
        # ticket resolution
        res = world.tickets[s["ticket"]]["resolutions"]
        if not res:
            fails.append("not_resolved")
        elif res[-1] != s["resolution"]:
            fails.append("wrong_resolution")
        elif len(res) > 1:
            fails.append("duplicate_resolve")
        # escalations
        n_esc = sum(1 for e in world.escalations if e["ticket_id"] == s["ticket"])
        if n_esc > s["escalations"]:
            fails.append("duplicate_escalation" if s["escalations"] else "wrong_escalation")
        elif n_esc < s["escalations"]:
            fails.append("missing_escalation")
        # addresses
        for oid, addr in s["addresses"].items():
            if world.orders[oid]["ship_address"] != addr:
                fails.append("wrong_address")
    # refunds or emails on customers that are not part of the task
    if any(r["order_id"] not in task_orders for r in world.refunds):
        fails.append("wrong_refund")
    task_customers = {s["customer"] for s in task["specs"]}
    if any(e["customer_id"] not in task_customers for e in world.emails):
        fails.append("wrong_email")
    return (len(fails) == 0, sorted(set(fails)))
