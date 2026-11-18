"""A small, deterministic customer-support world for an AI agent.

The world holds customers, orders, payments and tickets. Tools read and
change it. Every episode gets a fresh copy, so runs never affect each other.
Graders look only at the final state (refunds, emails, tickets, addresses),
never at what the agent says.
"""

import copy
import datetime as dt

TODAY = dt.date(2026, 10, 1)
REFUND_WINDOW_DAYS = 30
REFUND_CAP = 500.00

REFUND_POLICY = (
    f"Refund policy (today is {TODAY.isoformat()}):\n"
    f"1. Refunds are allowed only for orders placed within the last {REFUND_WINDOW_DAYS} days.\n"
    "2. Refund the exact price of each affected item. Issue one refund per item.\n"
    f"3. Any single refund over ${REFUND_CAP:.2f} must NOT be issued. Escalate it to a human instead.\n"
    "4. Duplicate charges: refund the duplicate payment amount once, regardless of order age.\n"
    "5. Shipping addresses can only be changed for orders with status 'processing'.\n"
    "Every ticket needs exactly one email to the customer and one resolve_ticket call."
)

RESOLUTION_CODES = [
    "STATUS_SHARED",
    "REFUNDED",
    "REFUND_DENIED",
    "ADDRESS_UPDATED",
    "ESCALATED",
    "DUPLICATE_CHARGE_REFUNDED",
    "NO_ISSUE_FOUND",
]

# Tools that change state. Faults on these can be "lost responses":
# the change happens, but the caller gets an error.
WRITE_TOOLS = {
    "issue_refund",
    "update_shipping_address",
    "send_email",
    "escalate_to_human",
    "resolve_ticket",
}


class BusinessError(Exception):
    """A normal, deterministic error (bad ID, rule broken). Never retried."""


READ_TOOLS = {"get_ticket", "find_customer", "list_orders", "get_order", "list_payments",
              "get_refund_policy", "list_refunds", "list_sent_emails"}


class World:
    def __init__(self, data):
        d = copy.deepcopy(data)
        self.customers = d["customers"]
        self.orders = d["orders"]
        self.payments = d["payments"]
        self.tickets = d["tickets"]
        self.refunds = []
        self.emails = []
        self.escalations = []
        self.address_changes = []
        self._idem = {}  # idempotency key -> cached result
        self._next = 1

    # ---------- helpers ----------
    def _id(self, prefix):
        self._next += 1
        return f"{prefix}-{self._next:04d}"

    def _order(self, order_id):
        if order_id not in self.orders:
            raise BusinessError(f"order {order_id!r} not found")
        return self.orders[order_id]

    def _order_total(self, o):
        return round(sum(i["price"] for i in o["items"]), 2)

    # ---------- dispatch ----------
    def call(self, tool, args, idempotency_key=None):
        if tool in WRITE_TOOLS and idempotency_key is not None:
            if idempotency_key in self._idem:
                return self._idem[idempotency_key]
        fn = getattr(self, "t_" + tool, None)
        if fn is None:
            raise BusinessError(f"unknown tool {tool!r}")
        try:
            result = fn(**args)
        except TypeError as e:
            raise BusinessError(f"bad arguments for {tool}: {e}")
        if tool in WRITE_TOOLS and idempotency_key is not None:
            self._idem[idempotency_key] = result
        return result

    # ---------- read tools ----------
    def t_get_ticket(self, ticket_id):
        t = self.tickets.get(ticket_id)
        if not t:
            raise BusinessError(f"ticket {ticket_id!r} not found")
        c = self.customers[t["customer_id"]]
        return {"ticket_id": ticket_id, "customer_email": c["email"],
                "subject": t["subject"], "body": t["body"], "status": t["status"]}

    def t_find_customer(self, email):
        for c in self.customers.values():
            if c["email"].lower() == str(email).lower():
                out = {"customer_id": c["id"], "name": c["name"], "email": c["email"]}
                if c.get("note"):
                    out["account_note"] = c["note"]
                return out
        raise BusinessError(f"no customer with email {email!r}")

    def t_list_orders(self, customer_id):
        if customer_id not in self.customers:
            raise BusinessError(f"customer {customer_id!r} not found")
        out = [{"order_id": o["id"], "date": o["date"], "status": o["status"],
                "total": self._order_total(o)}
               for o in self.orders.values() if o["customer_id"] == customer_id]
        return sorted(out, key=lambda x: x["order_id"])

    def t_get_order(self, order_id):
        o = self._order(order_id)
        return {"order_id": o["id"], "customer_id": o["customer_id"], "date": o["date"],
                "status": o["status"], "ship_address": o["ship_address"],
                "items": [{"name": i["name"], "price": i["price"]} for i in o["items"]],
                "total": self._order_total(o)}

    def t_list_payments(self, order_id):
        self._order(order_id)
        return self.payments.get(order_id, [])

    def t_get_refund_policy(self):
        return {"policy": REFUND_POLICY}

    # Observability tools: offered to the agent only in "observable" runs.
    def t_list_refunds(self, order_id):
        self._order(order_id)
        return [{"refund_id": r["refund_id"], "amount": r["amount"], "reason": r["reason"]}
                for r in self.refunds if r["order_id"] == order_id]

    def t_list_sent_emails(self, customer_id):
        if customer_id not in self.customers:
            raise BusinessError(f"customer {customer_id!r} not found")
        return [{"subject": e["subject"], "body": e["body"]}
                for e in self.emails if e["customer_id"] == customer_id]

    # ---------- write tools ----------
    def t_issue_refund(self, order_id, amount, reason):
        o = self._order(order_id)
        try:
            amount = round(float(amount), 2)
        except (TypeError, ValueError):
            raise BusinessError("amount must be a number")
        if amount <= 0:
            raise BusinessError("amount must be positive")
        paid = round(sum(p["amount"] for p in self.payments.get(order_id, [])), 2)
        refunded = round(sum(r["amount"] for r in self.refunds if r["order_id"] == order_id), 2)
        if refunded + amount > paid + 0.001:
            raise BusinessError(
                f"refund exceeds amount paid on {order_id} (paid {paid}, already refunded {refunded})")
        rid = self._id("R")
        self.refunds.append({"refund_id": rid, "order_id": order_id,
                             "amount": amount, "reason": str(reason)})
        return {"refund_id": rid, "order_id": order_id, "amount": amount}

    def t_update_shipping_address(self, order_id, address):
        o = self._order(order_id)
        if o["status"] != "processing":
            raise BusinessError(f"order {order_id} is {o['status']}; address can't be changed")
        o["ship_address"] = str(address)
        self.address_changes.append({"order_id": order_id, "address": str(address)})
        return {"order_id": order_id, "ship_address": o["ship_address"]}

    def t_send_email(self, customer_id, subject, body):
        if customer_id not in self.customers:
            raise BusinessError(f"customer {customer_id!r} not found")
        self.emails.append({"customer_id": customer_id, "subject": str(subject), "body": str(body)})
        return {"sent": True, "to": self.customers[customer_id]["email"]}

    def t_escalate_to_human(self, ticket_id, reason):
        if ticket_id not in self.tickets:
            raise BusinessError(f"ticket {ticket_id!r} not found")
        self.escalations.append({"ticket_id": ticket_id, "reason": str(reason)})
        return {"escalated": True, "ticket_id": ticket_id}

    def t_resolve_ticket(self, ticket_id, resolution_code):
        t = self.tickets.get(ticket_id)
        if not t:
            raise BusinessError(f"ticket {ticket_id!r} not found")
        if resolution_code not in RESOLUTION_CODES:
            raise BusinessError(f"resolution_code must be one of {RESOLUTION_CODES}")
        t["resolutions"].append(resolution_code)
        t["status"] = "resolved"
        return {"ticket_id": ticket_id, "status": "resolved"}


# ---------- tool schemas for the model ----------
def _s(props, req):
    return {"type": "object", "properties": props, "required": req}

STR = {"type": "string"}
NUM = {"type": "number"}

TOOL_SCHEMAS = [
    {"name": "get_ticket", "description": "Read a support ticket.",
     "input_schema": _s({"ticket_id": STR}, ["ticket_id"])},
    {"name": "find_customer", "description": "Look up a customer by email.",
     "input_schema": _s({"email": STR}, ["email"])},
    {"name": "list_orders", "description": "List a customer's orders (id, date, status, total).",
     "input_schema": _s({"customer_id": STR}, ["customer_id"])},
    {"name": "get_order", "description": "Get an order with its items, prices and shipping address.",
     "input_schema": _s({"order_id": STR}, ["order_id"])},
    {"name": "list_payments", "description": "List payments charged for an order.",
     "input_schema": _s({"order_id": STR}, ["order_id"])},
    {"name": "get_refund_policy", "description": "Read the refund and support policy.",
     "input_schema": _s({}, [])},
    {"name": "issue_refund", "description": "Refund an amount on an order. Moves real money.",
     "input_schema": _s({"order_id": STR, "amount": NUM, "reason": STR},
                        ["order_id", "amount", "reason"])},
    {"name": "update_shipping_address", "description": "Change the shipping address of a processing order.",
     "input_schema": _s({"order_id": STR, "address": STR}, ["order_id", "address"])},
    {"name": "send_email", "description": "Send an email to a customer.",
     "input_schema": _s({"customer_id": STR, "subject": STR, "body": STR},
                        ["customer_id", "subject", "body"])},
    {"name": "escalate_to_human", "description": "Hand a ticket to a human agent.",
     "input_schema": _s({"ticket_id": STR, "reason": STR}, ["ticket_id", "reason"])},
    {"name": "resolve_ticket", "description": "Close a ticket with a resolution code.",
     "input_schema": _s({"ticket_id": STR,
                         "resolution_code": {"type": "string", "enum": RESOLUTION_CODES}},
                        ["ticket_id", "resolution_code"])},
]


def schemas_with_keys():
    """Tool schemas for the key_available condition: every write tool gets an
    optional idempotency_key argument. The description documents it neutrally,
    as an API reference would; nothing tells the agent to use it."""
    out = []
    for t in TOOL_SCHEMAS:
        t = {**t, "input_schema": {**t["input_schema"],
                                   "properties": dict(t["input_schema"]["properties"])}}
        if t["name"] in WRITE_TOOLS:
            t["input_schema"]["properties"]["idempotency_key"] = {
                "type": "string",
                "description": "Optional. A unique string you choose for this operation. If a "
                               "request with the same key was already applied, it is not applied "
                               "again and the original result is returned."}
        out.append(t)
    return out


# Read tools that let the agent check what its earlier writes did. Offered only
# when a run sets observable=True (exploratory test of "an agent can only be
# careful about what it can observe").
OBSERVE_SCHEMAS = [
    {"name": "list_refunds", "description": "List refunds already issued on an order.",
     "input_schema": _s({"order_id": STR}, ["order_id"])},
    {"name": "list_sent_emails", "description": "List emails already sent to a customer.",
     "input_schema": _s({"customer_id": STR}, ["customer_id"])},
]


def tool_schemas(key_available=False, observable=False):
    """The tool list an agent sees for a run's settings."""
    base = schemas_with_keys() if key_available else [dict(t) for t in TOOL_SCHEMAS]
    return base + (OBSERVE_SCHEMAS if observable else [])
