"""
Sales Acceleration Digital Twin Simulation
==========================================
Domain: E-commerce / Retail
Author: Digital Twin Project
Description: A full-featured digital twin that simulates e-commerce sales pipeline,
             enables scenario testing, and generates KPI dashboards.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm
import random
import json
import os
from datetime import datetime, timedelta

# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SIMULATION_DAYS = 90          # Simulate 90 days
NUM_PRODUCTS = 6
NUM_CUSTOMERS = 500
BASE_DAILY_VISITORS = 1200


# ─────────────────────────────────────────────
# 2. PRODUCT CATALOGUE
# ─────────────────────────────────────────────
PRODUCTS = [
    {"id": "P001", "name": "Wireless Headphones",   "price": 79.99,  "cost": 32.00, "stock": 500, "category": "Electronics"},
    {"id": "P002", "name": "Running Shoes",          "price": 119.99, "cost": 45.00, "stock": 300, "category": "Apparel"},
    {"id": "P003", "name": "Smart Water Bottle",     "price": 34.99,  "cost": 12.00, "stock": 800, "category": "Fitness"},
    {"id": "P004", "name": "Yoga Mat",               "price": 49.99,  "cost": 18.00, "stock": 400, "category": "Fitness"},
    {"id": "P005", "name": "Portable Charger",       "price": 44.99,  "cost": 16.00, "stock": 600, "category": "Electronics"},
    {"id": "P006", "name": "Stainless Travel Mug",   "price": 29.99,  "cost": 10.00, "stock": 700, "category": "Lifestyle"},
]

# ─────────────────────────────────────────────
# 3. CUSTOMER SEGMENT MODEL
# ─────────────────────────────────────────────
SEGMENTS = {
    "Bargain Hunter":   {"size": 0.35, "price_sensitivity": 0.9, "loyalty": 0.2, "avg_basket": 1.2},
    "Loyal Shopper":    {"size": 0.25, "price_sensitivity": 0.3, "loyalty": 0.9, "avg_basket": 2.1},
    "Impulse Buyer":    {"size": 0.20, "price_sensitivity": 0.5, "loyalty": 0.4, "avg_basket": 1.5},
    "Premium Seeker":   {"size": 0.20, "price_sensitivity": 0.1, "loyalty": 0.6, "avg_basket": 1.8},
}


# ─────────────────────────────────────────────
# 4. DIGITAL TWIN CLASS
# ─────────────────────────────────────────────
class EcommerceSalesTwin:
    """
    Digital Twin of an E-commerce Retail Store.
    Mirrors the real store's sales pipeline: traffic → engagement → conversion → revenue.
    Supports scenario injection (promotions, stock changes, pricing).
    """

    def __init__(self, products, segments, base_visitors, days):
        self.products = [p.copy() for p in products]
        self.segments = segments
        self.base_visitors = base_visitors
        self.days = days
        self.history = []   # daily KPIs
        self.orders = []    # individual orders

    # ── Internal helpers ──────────────────────
    def _visitor_count(self, day, scenario):
        """Model daily visitors with weekly seasonality + scenario boosts."""
        weekday = day % 7
        seasonal = 1.0 + 0.15 * np.sin(2 * np.pi * weekday / 7)
        growth   = 1.0 + 0.003 * day                    # organic growth trend
        noise    = np.random.normal(1.0, 0.08)
        boost    = scenario.get("traffic_boost", 1.0)
        return max(int(self.base_visitors * seasonal * growth * noise * boost), 100)

    def _conversion_rate(self, product, segment_name, scenario):
        """Conversion depends on price sensitivity, discount, and segment loyalty."""
        seg = self.segments[segment_name]
        base_cr = 0.04
        discount = scenario.get("discount", 0.0)        # 0–1 fraction
        price_factor = 1.0 + seg["price_sensitivity"] * discount * 2
        loyalty_factor = 1.0 + seg["loyalty"] * 0.3
        stock_factor = min(product["stock"] / 100, 1.0) # drops if low stock
        noise = np.random.normal(1.0, 0.05)
        return min(base_cr * price_factor * loyalty_factor * stock_factor * noise, 0.25)

    def _basket_size(self, segment_name):
        """How many units per order."""
        seg = self.segments[segment_name]
        return max(1, int(np.random.poisson(seg["avg_basket"])))

    def _effective_price(self, product, scenario):
        """Apply discount from scenario."""
        discount = scenario.get("discount", 0.0)
        return round(product["price"] * (1 - discount), 2)

    # ── Main simulation loop ──────────────────
    def simulate(self, scenario=None):
        """Run the full simulation. scenario dict can override defaults."""
        if scenario is None:
            scenario = {}

        self.history = []
        self.orders  = []
        inventory    = {p["id"]: p["stock"] for p in self.products}

        start_date = datetime(2024, 1, 1)

        for day in range(self.days):
            date = start_date + timedelta(days=day)
            visitors = self._visitor_count(day, scenario)

            day_revenue = 0.0
            day_units   = 0
            day_orders  = 0
            day_margin  = 0.0

            for product in self.products:
                if inventory[product["id"]] <= 0:
                    continue

                eff_price = self._effective_price(product, scenario)

                for seg_name in self.segments:
                    seg = self.segments[seg_name]
                    seg_visitors = int(visitors * seg["size"])
                    cr = self._conversion_rate(product, seg_name, scenario)
                    buyers = int(seg_visitors * cr * (1 / NUM_PRODUCTS))

                    for _ in range(buyers):
                        qty = self._basket_size(seg_name)
                        qty = min(qty, inventory[product["id"]])
                        if qty <= 0:
                            break
                        revenue = eff_price * qty
                        margin  = (eff_price - product["cost"]) * qty
                        inventory[product["id"]] -= qty
                        day_revenue += revenue
                        day_margin  += margin
                        day_units   += qty
                        day_orders  += 1
                        self.orders.append({
                            "date": date.strftime("%Y-%m-%d"),
                            "day": day,
                            "product_id": product["id"],
                            "product_name": product["name"],
                            "segment": seg_name,
                            "qty": qty,
                            "unit_price": eff_price,
                            "revenue": revenue,
                            "margin": margin,
                        })

            # Restock trigger
            restock_threshold = scenario.get("restock_threshold", 50)
            for p in self.products:
                if inventory[p["id"]] < restock_threshold:
                    inventory[p["id"]] += int(p["stock"] * 0.5)

            day_cr = (day_orders / visitors * 100) if visitors else 0

            self.history.append({
                "day": day,
                "date": date.strftime("%Y-%m-%d"),
                "visitors": visitors,
                "orders": day_orders,
                "units_sold": day_units,
                "revenue": round(day_revenue, 2),
                "margin": round(day_margin, 2),
                "conversion_rate": round(day_cr, 3),
                "avg_order_value": round(day_revenue / day_orders, 2) if day_orders else 0,
            })

        return pd.DataFrame(self.history), pd.DataFrame(self.orders)


# ─────────────────────────────────────────────
# 5. SCENARIO RUNNER
# ─────────────────────────────────────────────
def run_scenarios(twin):
    """Run three scenarios: Baseline, Flash Sale, Premium Push."""
    scenarios = {
        "Baseline": {},
        "Flash Sale (20% off + 40% traffic boost)": {
            "discount": 0.20,
            "traffic_boost": 1.40,
            "restock_threshold": 80,
        },
        "Premium Push (5% off + loyalty focus)": {
            "discount": 0.05,
            "traffic_boost": 1.10,
        },
    }
    results = {}
    orders_map = {}
    for name, sc in scenarios.items():
        print(f"  Running scenario: {name} ...")
        hist, ords = twin.simulate(scenario=sc)
        results[name] = hist
        orders_map[name] = ords
    return results, orders_map


# ─────────────────────────────────────────────
# 6. KPI SUMMARY
# ─────────────────────────────────────────────
def print_kpi_summary(results):
    print("\n" + "="*65)
    print(f"{'SCENARIO KPI SUMMARY':^65}")
    print("="*65)
    fmt = "{:<40} {:>10} {:>12}"
    print(fmt.format("Scenario", "Revenue($)", "Conv.Rate(%)"))
    print("-"*65)
    for name, df in results.items():
        total_rev = df["revenue"].sum()
        avg_cr    = df["conversion_rate"].mean()
        print(fmt.format(name[:39], f"{total_rev:,.0f}", f"{avg_cr:.2f}"))
    print("="*65)


# ─────────────────────────────────────────────
# 7. DASHBOARD PLOTTING
# ─────────────────────────────────────────────
COLORS = {
    "Baseline": "#4A90D9",
    "Flash Sale (20% off + 40% traffic boost)": "#E74C3C",
    "Premium Push (5% off + loyalty focus)": "#2ECC71",
}

def plot_dashboard(results, orders_map, output_dir):
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor("#0F1117")
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── 1. Daily Revenue ──────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.set_facecolor("#1A1D27")
    for name, df in results.items():
        ax1.plot(df["day"], df["revenue"].rolling(7).mean(),
                 color=COLORS[name], linewidth=2.2, label=name)
    ax1.set_title("Daily Revenue (7-day Rolling Avg)", color="white", fontsize=12, pad=10)
    ax1.set_xlabel("Day", color="#AAB"); ax1.set_ylabel("Revenue ($)", color="#AAB")
    ax1.tick_params(colors="#AAB"); ax1.spines[:].set_color("#333")
    ax1.legend(fontsize=8, facecolor="#1A1D27", labelcolor="white")

    # ── 2. KPI Cards (text) ───────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor("#1A1D27"); ax2.axis("off")
    ax2.set_title("Total Revenue Comparison", color="white", fontsize=11, pad=8)
    y = 0.85
    for name, df in results.items():
        rev = df["revenue"].sum()
        ax2.text(0.05, y, f"{name.split('(')[0].strip()}", color=COLORS[name],
                 fontsize=9, fontweight="bold", transform=ax2.transAxes)
        ax2.text(0.05, y - 0.10, f"  ${rev:,.0f}", color="white",
                 fontsize=12, transform=ax2.transAxes)
        y -= 0.28

    # ── 3. Conversion Rate ────────────────────
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.set_facecolor("#1A1D27")
    for name, df in results.items():
        ax3.plot(df["day"], df["conversion_rate"].rolling(7).mean(),
                 color=COLORS[name], linewidth=2, label=name)
    ax3.set_title("Conversion Rate % (7-day Rolling Avg)", color="white", fontsize=12, pad=10)
    ax3.set_xlabel("Day", color="#AAB"); ax3.set_ylabel("Conv. Rate %", color="#AAB")
    ax3.tick_params(colors="#AAB"); ax3.spines[:].set_color("#333")
    ax3.legend(fontsize=8, facecolor="#1A1D27", labelcolor="white")

    # ── 4. Product Revenue Breakdown ──────────
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.set_facecolor("#1A1D27")
    baseline_orders = orders_map["Baseline"]
    prod_rev = baseline_orders.groupby("product_name")["revenue"].sum().sort_values()
    bars = ax4.barh(prod_rev.index, prod_rev.values, color="#4A90D9", edgecolor="#0F1117")
    ax4.set_title("Baseline Revenue by Product", color="white", fontsize=11, pad=8)
    ax4.tick_params(colors="#AAB", labelsize=8); ax4.spines[:].set_color("#333")
    ax4.set_xlabel("Revenue ($)", color="#AAB")
    for bar in bars:
        ax4.text(bar.get_width() + 100, bar.get_y() + bar.get_height()/2,
                 f"${bar.get_width():,.0f}", va="center", color="white", fontsize=7)

    # ── 5. Customer Segment Pie ────────────────
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.set_facecolor("#1A1D27")
    seg_rev = orders_map["Baseline"].groupby("segment")["revenue"].sum()
    wedge_colors = ["#4A90D9", "#E74C3C", "#2ECC71", "#F39C12"]
    ax5.pie(seg_rev.values, labels=seg_rev.index, autopct="%1.0f%%",
            colors=wedge_colors, textprops={"color": "white", "fontsize": 8})
    ax5.set_title("Revenue by Segment (Baseline)", color="white", fontsize=11, pad=8)

    # ── 6. Cumulative Revenue ─────────────────
    ax6 = fig.add_subplot(gs[2, 1:])
    ax6.set_facecolor("#1A1D27")
    for name, df in results.items():
        ax6.plot(df["day"], df["revenue"].cumsum() / 1000,
                 color=COLORS[name], linewidth=2.2, label=name)
    ax6.set_title("Cumulative Revenue Over 90 Days ($ thousands)", color="white", fontsize=12, pad=10)
    ax6.set_xlabel("Day", color="#AAB"); ax6.set_ylabel("Revenue ($ K)", color="#AAB")
    ax6.tick_params(colors="#AAB"); ax6.spines[:].set_color("#333")
    ax6.legend(fontsize=8, facecolor="#1A1D27", labelcolor="white")

    fig.suptitle("Sales Acceleration Digital Twin — E-Commerce Dashboard",
                 color="white", fontsize=16, fontweight="bold", y=0.98)

    out_path = os.path.join(output_dir, "dashboard.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  Dashboard saved → {out_path}")
    return out_path


# ─────────────────────────────────────────────
# 8. EXPORT DATA
# ─────────────────────────────────────────────
def export_data(results, orders_map, output_dir):
    for name, df in results.items():
        safe = name.split("(")[0].strip().replace(" ", "_").lower()
        df.to_csv(os.path.join(output_dir, f"kpi_{safe}.csv"), index=False)
    orders_map["Baseline"].to_csv(os.path.join(output_dir, "orders_baseline.csv"), index=False)
    print(f"  CSV exports saved to {output_dir}/")


# ─────────────────────────────────────────────
# 9. MAIN
# ─────────────────────────────────────────────
def main():
    print("\n╔══════════════════════════════════════════════════╗")
    print("║   Sales Acceleration Digital Twin Simulation     ║")
    print("║   Domain: E-Commerce / Retail                    ║")
    print("╚══════════════════════════════════════════════════╝\n")

    print("[1/4] Initialising Digital Twin ...")
    twin = EcommerceSalesTwin(
        products=PRODUCTS,
        segments=SEGMENTS,
        base_visitors=BASE_DAILY_VISITORS,
        days=SIMULATION_DAYS,
    )

    print("[2/4] Running Scenarios ...")
    results, orders_map = run_scenarios(twin)

    print("[3/4] Generating KPI Summary ...")
    print_kpi_summary(results)

    print("\n[4/4] Plotting Dashboard ...")
    plot_dashboard(results, orders_map, OUTPUT_DIR)

    export_data(results, orders_map, OUTPUT_DIR)

    print("\n✅ Simulation complete! All outputs saved to ./outputs/\n")


if __name__ == "__main__":
    main()
