"""
DAS Trader CMD-API execution backend — SIMULATED trades only.

Closes the core gap: the dashboard's PaperTradingBot is in-memory; this routes orders to a
real DAStrader instance over its CMD socket so Falcon becomes an actual automated trading
system. SIM account only.

SAFETY (non-negotiable):
  * SIM ONLY — refuses to operate on the LIVE account (1RB16917). See memory das-pos-type,
    check-account-column-first.
  * DRY-RUN by default — place_order() formats and returns the command WITHOUT sending
    unless BOTH the instance is created live=True AND env FALCON_DAS_LIVE=1. The DAS CMD
    NEWORDER syntax is not yet manual-verified (memory: falcon-auto-trader "hotkey syntax
    pending DAS manual review"), so live firing stays opt-in until that is confirmed.
  * Routes: limit=REB25L, market=REB25M (memory das-default-routes). A bare "S"/SMRTL can
    open a short (memory das-sell-opens-short) — we always send an explicit route.

Connection: DAS binds loopback; from a container use --network=host or host gateway.
"""
import os
import socket
import time
import logging

logger = logging.getLogger(__name__)

LIVE_ACCOUNT = "1RB16917"            # never trade this here
LIMIT_ROUTE = "REB25L"
MARKET_ROUTE = "REB25M"
_TOKEN = [0]   # client-side NEWORDER token counter (DAS echoes it back in %ORDER)


class DASExecutionClient:
    def __init__(self, host=None, port=None, user=None, password=None, account=None,
                 live=False, timeout=4.0):
        self.host = host or os.getenv("DAS_HOST", "127.0.0.1")
        self.port = int(port or os.getenv("DAS_PORT", "3192"))
        self.user = user or os.getenv("DAS_USER", "")
        self._password = password or os.getenv("DAS_PASSWORD", "")
        self.account = account or os.getenv("DAS_ACCOUNT", "")
        # live firing requires BOTH the constructor flag and the env gate
        self.live = bool(live) and os.getenv("FALCON_DAS_LIVE") == "1"
        self.timeout = timeout
        self.sock = None
        self.logged_in = False
        if self.account == LIVE_ACCOUNT:
            raise ValueError(f"refusing to use LIVE account {LIVE_ACCOUNT}; SIM only")

    # ---- connection ----
    def connect(self):
        self.sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        self.sock.settimeout(self.timeout)
        return self

    def _send(self, line):
        self.sock.sendall((line + "\r\n").encode())

    def _read(self, settle=0.6):
        """Read whatever DAS streams back within the settle window."""
        buf = b""
        end = time.time() + settle
        self.sock.settimeout(0.4)
        while time.time() < end:
            try:
                chunk = self.sock.recv(4096)
                if not chunk:
                    break
                buf += chunk
                end = time.time() + 0.3   # extend on activity
            except socket.timeout:
                break
        return buf.decode(errors="replace")

    def login(self):
        self._send(f"LOGIN {self.user} {self._password} {self.account}")
        resp = self._read(1.2)
        self.logged_in = ("success" in resp.lower()) or ("logged in" in resp.lower()) \
            or ("#login" in resp.lower() and "fail" not in resp.lower() and "error" not in resp.lower())
        logger.info("DAS login %s (acct %s)", "OK" if self.logged_in else "FAILED", self.account)
        return self.logged_in, resp.strip()

    # ---- read-only queries (safe) ----
    def buying_power(self):
        self._send("GET BP")
        return self._read().strip()

    def positions(self):
        # DAS CMD position query (POS alone is rejected; POSREFRESH streams #POS..#POSEND)
        self._send("POSREFRESH")
        return self._read().strip()

    def orders(self):
        self._send("GET ORDERS")
        return self._read().strip()

    # ---- order construction + (guarded) send ----
    def _next_token(self):
        _TOKEN[0] += 1
        return _TOKEN[0]

    def build_order(self, symbol, side, shares, price=None, route=None, tif="DAY",
                    stop_type=None, stop_price=None, stop_price2=None, token=None):
        """DAS CMD-API NEWORDER — AUTHORITATIVE positional form (CMD API Manual, Tim Bian):
            NEWORDER <token> <b/s> <symbol> <route> <share> <price|MKT|STOP..> [TIF=...]
          Limit:    NEWORDER 1 B MSFT ARCA 100 200.5 TIF=DAY+
          Market:   NEWORDER 2 S MSFT SMAT 100 MKT TIF=DAY
          StopMkt:  NEWORDER 4 S MSFT SMAT 100 STOPMKT <stop> TIF=DAY
          StopLmt:  NEWORDER 5 B MSFT SMAT 100 STOPLMT <stop> <limit> TIF=DAY
        side: B/S/SS (server reconciles S vs SS by position); BUY/SELL/SHORT/COVER mapped.
        token: client-set integer to trace the order (DAS returns it in %ORDER). Default route:
        market REB25M, limit REB25L (memory das-default-routes)."""
        side = {"BUY": "B", "SELL": "S", "SHORT": "SS", "COVER": "BC"}.get(side.upper(), side.upper())
        if side not in ("B", "S", "SS", "BC", "BO", "SO", "SC"):
            raise ValueError("side must be B/S/SS/BC (or BUY/SELL/SHORT/COVER)")
        if token is None:
            token = self._next_token()
        if route is None:
            route = LIMIT_ROUTE if (price is not None or stop_type) else MARKET_ROUTE
        parts = ["NEWORDER", str(token), side, symbol.upper(), route, str(int(shares))]
        if stop_type:
            parts.append(stop_type.upper())                       # STOPMKT/STOPLMT/STOPTRAILING/STOPRANGE
            if stop_price is not None:
                parts.append(f"{float(stop_price):.2f}")
            if stop_price2 is not None:                           # STOPLMT limit px / STOPRANGE high px
                parts.append(f"{float(stop_price2):.2f}")
        elif price is None:
            parts.append("MKT")                                   # market keyword (not a 0 price)
        else:
            parts.append(f"{float(price):.2f}")
        parts.append(f"TIF={tif}")
        return " ".join(parts)

    def place_order(self, symbol, side, shares, price=None, route=None):
        cmd = self.build_order(symbol, side, shares, price, route)
        if not self.live:
            logger.warning("DRY-RUN (not sent): %s", cmd)
            return {"status": "dry_run", "command": cmd, "sent": False,
                    "note": "set live=True + FALCON_DAS_LIVE=1 to fire; verify NEWORDER syntax first"}
        if not self.logged_in:
            return {"status": "error", "error": "not logged in", "command": cmd, "sent": False}
        self._send(cmd)
        ack = self._read().strip()
        return {"status": "sent", "command": cmd, "sent": True, "ack": ack}

    def close(self):
        try:
            if self.sock:
                self.sock.close()
        finally:
            self.sock = None
            self.logged_in = False


def health_check():
    """Connect + login + read BP/POS (no orders). Returns a dict for /api/das/health."""
    c = DASExecutionClient()
    try:
        c.connect()
        ok, resp = c.login()
        out = {"connected": True, "account": c.account, "login_ok": ok,
               "live_firing": c.live}
        if ok:
            out["buying_power"] = c.buying_power()[:400]
            out["positions"] = c.positions()[:400]
        else:
            out["login_response"] = resp[:400]
        return out
    except Exception as e:
        return {"connected": False, "error": str(e)}
    finally:
        c.close()


if __name__ == "__main__":
    import json
    print(json.dumps(health_check(), indent=2))
