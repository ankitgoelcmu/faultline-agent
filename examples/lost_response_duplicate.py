"""A write commits, its response is lost, and a naive retry repeats it.

  python examples/lost_response_duplicate.py

Standalone (standard library only): copy it and replace `Store.refund`
with a call to your own tool wrapper. Expected output ends with "OK".
"""


class Timeout(Exception):
    pass


class Store:
    def __init__(self):
        self.refunds = []
        self.seen = {}               # idempotency key -> first result

    def refund(self, order_id, amount, key=None, lose_response=False):
        if key is not None and key in self.seen:
            return self.seen[key]    # deduplicated: applied once, same result
        self.refunds.append((order_id, amount))
        result = {"refund_id": f"R{len(self.refunds)}"}
        if key is not None:
            self.seen[key] = result
        if lose_response:
            raise Timeout("write committed, response lost")
        return result


def call_with_retries(store, use_key, attempts=3):
    key = "refund:O-1:25.00" if use_key else None   # derived from the operation
    for i in range(attempts):
        try:
            return store.refund("O-1", 25.00, key=key, lose_response=(i == 0))
        except Timeout:
            continue


naive, keyed = Store(), Store()
call_with_retries(naive, use_key=False)
call_with_retries(keyed, use_key=True)
print(f"naive retry:     {len(naive.refunds)} refunds (one intended)")
print(f"retry with key:  {len(keyed.refunds)} refund")
assert len(naive.refunds) == 2 and len(keyed.refunds) == 1
print("OK")
