"""An idempotency key that is too broad silently drops a legitimate write.

  python examples/broad_key_merge.py

Standalone (standard library only). Keying an email by recipient alone
merges two different emails to the same customer; keying by the whole
operation keeps both. Expected output ends with "OK".
"""

import hashlib
import json


def key(op, fields):
    return hashlib.sha256(json.dumps({k: op[k] for k in fields}, sort_keys=True).encode()).hexdigest()


def send_all(ops, fields):
    sent, seen = [], set()
    for op in ops:
        k = key(op, fields)
        if k in seen:
            continue                 # treated as a retry and dropped
        seen.add(k)
        sent.append(op)
    return sent


ops = [{"to": "c-17", "subject": "Your refund", "body": "Issued today."},
       {"to": "c-17", "subject": "Survey", "body": "How did we do?"}]
broad = send_all(ops, ["to"])
full = send_all(ops, ["to", "subject", "body"])
print(f"key = recipient:           {len(broad)} of 2 emails sent")
print(f"key = recipient+subject+body: {len(full)} of 2 emails sent")
assert len(broad) == 1 and len(full) == 2
print("OK")
