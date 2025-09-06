def round_price(price, tick_size=0.01):
    return round(price / tick_size) * tick_size

def format_pnl(pnl):
    return f"${pnl:,.2f}"

def safe_divide(a, b):
    return a / b if b else 0.0 