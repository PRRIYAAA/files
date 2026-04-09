import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from collections import Counter
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="Smart Suggestion System", page_icon="💻",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

:root {
    --primary: #4F46E5;
    --primary-light: #818CF8;
    --primary-dark: #3730A3;
    --accent: #7C3AED;
    --bg-main: #F8FAFC;
    --glass: rgba(255, 255, 255, 0.72);
    --border: rgba(226, 232, 240, 0.8);
    --text-main: #1E293B;
    --text-muted: #64748B;
    --shadow: 0 4px 6px -1px rgb(0 0 0 / 0.05), 0 2px 4px -2px rgb(0 0 0 / 0.05);
}

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

.stApp {
    background: radial-gradient(circle at top right, #F1F5F9, #F8FAFC);
}

#MainMenu, footer, header {visibility: hidden;}

.block-container {
    padding: 2rem 3rem 4rem;
    max-width: 1350px;
}

/* Glassmorphism Containers */
section[data-testid="stSidebar"] {
    background: var(--glass);
    backdrop-filter: blur(12px);
    border-right: 1px solid var(--border);
}

.stTabs [data-baseweb="tab-list"] {
    background: var(--glass);
    border-radius: 14px;
    padding: 6px;
    gap: 8px;
    border: 1px solid var(--border);
    box-shadow: var(--shadow);
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 10px;
    color: var(--text-muted);
    font-size: 14px;
    font-weight: 500;
    padding: 10px 24px;
    border: none;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%) !important;
    color: #FFFFFF !important;
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
}

/* Premium Card Styles */
.premium-card {
    background: var(--glass);
    backdrop-filter: blur(8px);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 20px;
    transition: all 0.3s ease;
    box-shadow: var(--shadow);
}

.premium-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    border-color: var(--primary-light);
}

/* Sidebar Analytics Panel */
.sidebar-stat {
    background: #FFFFFF;
    border-radius: 12px;
    padding: 12px 15px;
    margin-bottom: 12px;
    border: 1px solid var(--border);
    transition: all 0.2s ease;
}

.sidebar-stat:hover {
    border-color: var(--primary-light);
    transform: translateX(4px);
}

[data-testid="stMetricValue"] {
    font-size: 28px !important;
    font-weight: 700 !important;
    color: var(--text-main) !important;
    letter-spacing: -0.01em;
}

[data-testid="stMetricLabel"] {
    font-size: 13px !important;
    color: var(--text-muted) !important;
    font-weight: 500 !important;
}

hr {
    margin: 1.5em 0 !important;
    opacity: 0.1;
}
</style>""", unsafe_allow_html=True)

# ── helpers ──────────────────────────────────────────────────
def count_items(series, excludes=None):
    excludes = excludes or ['None','It depends','warranty','']
    c = Counter()
    for v in series.fillna(''):
        for x in str(v).split(','):
            x = x.strip()
            if x and x not in excludes: c[x] += 1
    return c

def insight_box(text, color="#4F46E5", bg="#F4F7FF", border="#C7D2FE"):
    st.markdown(f"""<div style="background:{bg};border-left:4px solid {color};
        border-radius:4px 12px 12px 4px;padding:16px 20px;margin-bottom:15px;
        font-size:14px;color:#334155;line-height:1.6;box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        border:1px solid {border}; border-left-width: 4px;">{text}</div>""",
        unsafe_allow_html=True)

def ui_card(title, value, subtext=None, icon="📈", gradient=False):
    bg_style = "linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%)" if gradient else "white"
    text_color = "white" if gradient else "#1E293B"
    sub_color = "rgba(255,255,255,0.8)" if gradient else "#64748B"
    border_style = "none" if gradient else "1px solid rgba(226, 232, 240, 0.8)"
    
    st.markdown(f"""
    <div class="premium-card" style="background:{bg_style}; color:{text_color}; border:{border_style};">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
            <span style="font-size:12px; font-weight:600; text-transform:uppercase; letter-spacing:0.05em; opacity:0.9;">{title}</span>
            <span style="font-size:20px;">{icon}</span>
        </div>
        <div style="font-size:28px; font-weight:700; margin-bottom:4px;">{value}</div>
        {f'<div style="font-size:12px; color:{sub_color};">{subtext}</div>' if subtext else ''}
    </div>
    """, unsafe_allow_html=True)

def section_label(text):
    st.markdown(f"""<div style="font-size:12px;font-weight:700;letter-spacing:0.1em;
        text-transform:uppercase;color:#94A3B8;margin-bottom:12px;margin-top:5px;
        display:flex;align-items:center;gap:8px;">
        <div style="width:12px;height:2px;background:linear-gradient(90deg, #4F46E5, transparent);border-radius:2px;"></div>
        {text}</div>""",
        unsafe_allow_html=True)

# ── data loading ─────────────────────────────────────────────
@st.cache_data
def load_laptop_data():
    df = pd.read_csv("cleaned_dataset.csv")
    df['Items_List'] = df['Included_Items'].fillna('').apply(
        lambda x: [i.strip() for i in str(x).split('|')
                   if i.strip() and i.strip().lower() != 'none'])
    return df

@st.cache_resource
def generate_rules(df):
    te = TransactionEncoder()
    te_data = te.fit(df['Items_List'].tolist()).transform(df['Items_List'].tolist())
    df_trans = pd.DataFrame(te_data, columns=te.columns_)
    freq  = apriori(df_trans, min_support=0.01, use_colnames=True)
    rules = association_rules(freq, metric="confidence", min_threshold=1.0)
    return rules[rules["lift"] >= 1].reset_index(drop=True)

@st.cache_data
def load_survey():
    df = pd.read_csv("Survey-Response-500.csv")
    df.columns = [c.strip() for c in df.columns]
    return df

def classify_user(row):
    if row["RAM_GB"] >= 16 and row["Final_Price"] > 70000: return "Premium"
    elif row["RAM_GB"] >= 8: return "Mid"
    return "Low"

df_laptop    = load_laptop_data()
rules        = generate_rules(df_laptop)
survey       = load_survey()
N            = len(survey)
tier_counts  = df_laptop.apply(classify_user, axis=1).value_counts()

# ── cost model (in code only, not exposed as UI config) ──────
# Total cost per unit = product_cost + personnel_cost + campaign_cost
COST_MODEL = {
    "Wireless Mouse":         {"price":1200,  "product_cost":700,   "personnel_cost":50,  "campaign_cost":80},
    "Laptop Bag":             {"price":2500,  "product_cost":1400,  "personnel_cost":60,  "campaign_cost":100},
    "Laptop Stand":           {"price":2000,  "product_cost":1100,  "personnel_cost":50,  "campaign_cost":90},
    "Cooling Pad":            {"price":1500,  "product_cost":850,   "personnel_cost":50,  "campaign_cost":80},
    "External Storage/SSD":   {"price":5000,  "product_cost":3200,  "personnel_cost":80,  "campaign_cost":120},
    "Processor Upgrade (i7)": {"price":12000, "product_cost":9000,  "personnel_cost":200, "campaign_cost":300},
    "Battery Replacement":    {"price":3000,  "product_cost":1800,  "personnel_cost":150, "campaign_cost":150},
    "RAM Upgrade (16GB)":     {"price":4000,  "product_cost":2800,  "personnel_cost":100, "campaign_cost":150},
    "Storage (NVMe SSD)":     {"price":5000,  "product_cost":3500,  "personnel_cost":100, "campaign_cost":150},
    "GPU Upgrade":            {"price":25000, "product_cost":20000, "personnel_cost":400, "campaign_cost":500},
}
CONVERSION = 0.20

# (Global survey calculations moved to Tab 2 for dynamic filtering)

# ── sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""<div style="margin-bottom:24px;">
      <div style="font-size:22px;font-weight:700;color:#1E293B;letter-spacing:-0.03em;">
        <span style="color:#4F46E5;">Smart</span>Suggest</div>
      <div style="font-size:12px;color:#94A3B8;margin-top:2px;font-weight:500;">HP Laptop Intelligence Engine</div>
    </div>""", unsafe_allow_html=True)
    
    st.divider()
    
    section_label("System Context")
    stats = [
        ("Survey Respondents", f"{N:,}", "👤"),
        ("Laptop Catalog", f"{len(df_laptop):,}", "💻"),
        ("Association Rules", f"{len(rules)}", "🔗"),
        ("Conversion Target", "20%", "🎯")
    ]
    
    for label, val, icon in stats:
        st.markdown(f"""<div class="sidebar-stat">
            <div style="font-size:11px; color:#94A3B8; font-weight:600; text-transform:uppercase; margin-bottom:2px;">{label}</div>
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:16px; font-weight:700; color:#1E293B;">{val}</span>
                <span style="font-size:14px;">{icon}</span>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:15px'></div>", unsafe_allow_html=True)
    section_label("User Tier Distribution")
    for tier, color, emoji in [("Premium","#7C3AED","💎"), ("Mid","#2563EB","🚀"), ("Low","#059669","📌")]:
        count = tier_counts.get(tier, 0)
        pct = count / len(df_laptop) * 100
        st.markdown(f"""<div style="margin-bottom:14px; padding:0 4px;">
            <div style="display:flex; justify-content:space-between; margin-bottom:6px; align-items:center;">
                <span style="font-size:13px; font-weight:600; color:#475569;">{emoji} {tier}</span>
                <span style="font-size:12px; color:#94A3B8; font-weight:500;">{pct:.0f}%</span>
            </div>
            <div style="background:#F1F5F9; border-radius:10px; height:6px; overflow:hidden;">
                <div style="background:linear-gradient(90deg, {color}, {color}BB); width:{pct:.0f}%; height:100%; border-radius:10px;"></div>
            </div>
        </div>""", unsafe_allow_html=True)
    st.divider()
    st.markdown('<div style="font-size:11px;color:#C0C4D0;text-align:center;">MOVATE Internship · Week 6</div>',
                unsafe_allow_html=True)

# ── page header ───────────────────────────────────────────────
st.markdown("""<div style="margin-bottom:30px;">
  <h1 style="font-size:36px;font-weight:700;color:#1E293B;margin:0;letter-spacing:-0.04em;">
    Intelligent Suggestion <span style="color:#4F46E5;">System</span></h1>
  <p style="font-size:15px;color:#64748B;margin:8px 0 0;font-weight:400;">
    Advanced Apriori Engine &nbsp;·&nbsp; Hardware Optimization &nbsp;·&nbsp;
    500-User Sentiment Overlay &nbsp;·&nbsp; Business Profitability Model</p>
</div>""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([
    "💡  Recommendations",
    "📊  Survey Overlay & Business Analysis",
    "🔍  Association Rules",
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════
with tab1:
    selected_name = st.selectbox("Select a laptop from the HP catalog",
                                 options=df_laptop['Name'].unique())
    laptop = df_laptop[df_laptop['Name'] == selected_name].iloc[0]
    tier   = classify_user(laptop)
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    tier_cfg = {
        "Premium": ("#7C3AED","linear-gradient(135deg, #F5F3FF, #EDE9FE)","#DDD6FE","💎","Platinum Performance Category"),
        "Mid":     ("#2563EB","linear-gradient(135deg, #EFF6FF, #DBEAFE)","#BFDBFE","🚀","Standard Performance Category"),
        "Low":     ("#059669","linear-gradient(135deg, #F0FDF4, #DCFCE7)","#BBF7D0","📌","Entry Performance Category"),
    }
    tc = tier_cfg[tier]
    st.markdown(f"""
    <div style="background:{tc[1]}; border:1px solid {tc[2]}; border-radius:16px;
        padding:20px 24px; margin-bottom:28px; display:flex; align-items:center; gap:18px; 
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);">
      <div style="background:white; width:56px; height:56px; border-radius:12px; display:flex; 
          align-items:center; justify-content:center; font-size:28px; box-shadow:0 2px 4px rgba(0,0,0,0.05);">{tc[3]}</div>
      <div>
        <div style="font-size:12px; font-weight:700; color:{tc[0]}; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:4px;">{tc[4]}</div>
        <div style="font-size:18px; font-weight:700; color:#1E293B; margin-bottom:4px;">{selected_name}</div>
        <div style="font-size:13px; color:#64748B; font-weight:500;">
          {laptop['RAM_GB']}GB RAM &nbsp;·&nbsp; {laptop['Storage_GB']}GB Storage
          &nbsp;·&nbsp; {laptop['Processor_Level']} &nbsp;·&nbsp; <span style="color:#1E293B; font-weight:600;">₹{laptop['Final_Price']:,.0f}</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col_up, col_cross = st.columns(2)
    with col_up:
        section_label("📈 Upsell — Hardware Optimization")
        if tier == "Premium":
            cards = [("✨","#7C3AED","#F5F3FF","#DDD6FE","Top-tier system","Hardware meets all professional standards"),
                     ("🎯","#7C3AED","#F5F3FF","#DDD6FE","Focus on accessories","Maximise productivity with peripherals")]
        elif tier == "Mid":
            cards = []
            if laptop['RAM_GB'] < 16:
                cards.append(("🧠","#2563EB","#EFF6FF","#BFDBFE","RAM Upgrade → 16GB","PRO-TIP: Helps in heavy multitasking"))
            cards.append(("⚡","#2563EB","#EFF6FF","#BFDBFE","Processor → i7 / Ryzen 7","Better for creative workloads"))
            if laptop['Storage_GB'] < 512:
                cards.append(("💾","#2563EB","#EFF6FF","#BFDBFE","Storage → 512 GB+","Recommended for modern apps"))
        else:
            cards = [("⚡","#059669","#F0FDF4","#BBF7D0","RAM Upgrade → 16GB","URGENT — 2× performance boost"),
                     ("💾","#059669","#F0FDF4","#BBF7D0","Switch to NVMe SSD","UPGRADE — 5× faster boot times"),
                     ("🔋","#059669","#F0FDF4","#BBF7D0","Battery Replacement","Extend daily usage time")]
        for emoji, color, bg, border, title, sub in cards:
            st.markdown(f"""<div class="premium-card" style="padding:18px; margin-bottom:14px; border-left:4px solid {color};">
              <div style="display:flex; align-items:flex-start; gap:14px;">
                <div style="background:{bg}; width:38px; height:38px; border-radius:10px; display:flex; 
                   align-items:center; justify-content:center; font-size:20px; flex-shrink:0; border:1px solid {border};">{emoji}</div>
                <div>
                  <div style="font-size:14px; font-weight:700; color:#1E293B; margin-bottom:2px;">{title}</div>
                  <div style="font-size:12px; color:#64748B; line-height:1.5; font-weight:400;">{sub}</div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

    with col_cross:
        section_label("🎁 Cross-sell — Recommended Accessories")
        current_acc = list(laptop['Items_List']); suggested = []
        if not rules.empty:
            for _, rule in rules.sort_values("confidence", ascending=False).iterrows():
                if set(rule['antecedents']).issubset(set(current_acc)):
                    for item in rule['consequents']:
                        if item not in current_acc and item not in suggested:
                            suggested.append(item)
        if not suggested:
            suggested = ["HP Wireless Mouse (High Comfort)","HP Premium Laptop Bag","Multi-port USB-C Hub"]
        icon_map  = {"mouse":"🖱️","bag":"🎒","backpack":"🎒","hub":"🔌","office":"📄",
                     "365":"📄","xbox":"🎮","keyboard":"⌨️","stand":"🪜","ssd":"📀","hard":"📀","cooling":"🌬️"}
        color_map = {"mouse":"#4F46E5","bag":"#059669","backpack":"#059669","hub":"#D97706",
                     "office":"#2563EB","365":"#2563EB","xbox":"#16A34A","keyboard":"#7C3AED",
                     "stand":"#EA580C","ssd":"#0891B2","hard":"#0891B2","cooling":"#0891B2"}
        def get_ic(name):
            nl = name.lower()
            for k in icon_map:
                if k in nl: return icon_map[k], color_map[k]
            return "📦", "#6B7280"
        for i, item in enumerate(suggested[:3], 1):
            icon, color = get_ic(item)
            short = item[:65] + "…" if len(item) > 65 else item
            st.markdown(f"""<div class="premium-card" style="padding:16px; margin-bottom:14px;">
              <div style="display:flex; align-items:center; gap:16px;">
                <div style="width:48px; height:48px; background:{color}10; border-radius:12px;
                   display:flex; align-items:center; justify-content:center;
                   font-size:22px; border:1px solid {color}20; flex-shrink:0; color:{color};">{icon}</div>
                <div style="flex:1;">
                   <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:2px;">
                      <span style="font-size:10px; font-weight:700; color:{color}; text-transform:uppercase; letter-spacing:0.06em;">Priority Suggestion</span>
                      <span style="font-size:10px; color:#94A3B8; font-weight:600;">HP Original</span>
                   </div>
                   <div style="font-size:14px; color:#334155; font-weight:600; line-height:1.4;">{short}</div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 2 — SURVEY OVERLAY & BUSINESS ANALYSIS
# ══════════════════════════════════════════════════════════════
with tab2:
    # ── FILTERS ───────────────────────────────
    st.markdown("### 🔎 Filters")

    col_f1, col_f2 = st.columns(2)

    with col_f1:
        age_filter = st.multiselect(
            "Select Age Group",
            options=survey["Age Group"].dropna().unique(),
            default=survey["Age Group"].dropna().unique()
        )

    with col_f2:
        occ_filter = st.multiselect(
            "Select Occupation",
            options=survey["Occupation"].dropna().unique(),
            default=survey["Occupation"].dropna().unique()
        )

    # Apply filters
    filtered_survey = survey[
        (survey["Age Group"].isin(age_filter)) &
        (survey["Occupation"].isin(occ_filter))
    ]

    N_filtered = len(filtered_survey)

    # ── CALCULATIONS (Dynamic based on filters) ──
    want_c = count_items(filtered_survey['Which accessories would you consider buying...'] if 'Which accessories would you consider buying...' in filtered_survey.columns else filtered_survey['Which accessories would you consider buying with a new laptop?'])
    curr_c = count_items(filtered_survey['Which accessories do you currently use ?'])
    upg_c  = filtered_survey['If you were to upgrade your laptop, which feature would you prefer to improve?'].value_counts().to_dict()

    ACC_MAP = [
        ("Wireless Mouse",       "Wireless Mouse",            "Wireless Mouse"),
        ("Laptop Bag",           "Laptop Bag",                "Laptop Bag"),
        ("Laptop Stand",         "Laptop Stand",              "Laptop Stand"),
        ("Cooling Pad",          "Cooling Pad",               "Cooling Pad"),
        ("External Storage/SSD", "External Hard Drive / SSD", "External Storage"),
    ]
    acc_rows = []
    for name, own_key, want_key in ACC_MAP:
        cm   = COST_MODEL[name]
        own  = curr_c.get(own_key, 0)
        want = want_c.get(want_key, 0)
        opp  = min(want, N_filtered - own)
        b    = round(opp * CONVERSION, 1)
        rev  = int(b * cm["price"])
        tc   = int(b * (cm["product_cost"] + cm["personnel_cost"] + cm["campaign_cost"]))
        gp   = rev - tc
        acc_rows.append(dict(name=name, own=own, want=want, opp=opp, buyers=b,
                             price=cm["price"], revenue=rev, total_cost=tc, gross_profit=gp,
                             product_cost=int(b*cm["product_cost"]),
                             personnel_cost=int(b*cm["personnel_cost"]),
                             campaign_cost=int(b*cm["campaign_cost"])))
    df_acc = pd.DataFrame(acc_rows)

    UPG_MAP = [
        ("Processor Upgrade (i7)", "Processor performance"),
        ("Battery Replacement",    "Battery life"),
        ("RAM Upgrade (16GB)",     "RAM capacity"),
        ("Storage (NVMe SSD)",     "Storage capacity"),
        ("GPU Upgrade",            "Graphics performance"),
    ]
    upg_rows = []
    for name, survey_key in UPG_MAP:
        cm   = COST_MODEL[name]
        hits = upg_c.get(survey_key, 0)
        b    = round(hits * CONVERSION, 1)
        rev  = int(b * cm["price"])
        tc   = int(b * (cm["product_cost"] + cm["personnel_cost"] + cm["campaign_cost"]))
        gp   = rev - tc
        upg_rows.append(dict(name=name, hits=hits, buyers=b,
                             price=cm["price"], revenue=rev, total_cost=tc, gross_profit=gp,
                             product_cost=int(b*cm["product_cost"]),
                             personnel_cost=int(b*cm["personnel_cost"]),
                             campaign_cost=int(b*cm["campaign_cost"])))
    df_up = pd.DataFrame(upg_rows)

    total_acc_rev  = int(df_acc["revenue"].sum())
    total_upg_rev  = int(df_up["revenue"].sum())
    total_revenue  = total_acc_rev + total_upg_rev
    total_acc_cost = int(df_acc["total_cost"].sum())
    total_upg_cost = int(df_up["total_cost"].sum())
    total_cost_all = total_acc_cost + total_upg_cost
    total_profit   = total_revenue - total_cost_all
    roi            = round((total_profit / total_cost_all * 100) if total_cost_all > 0 else 0, 1)

    st.markdown(f"""<div style="background:#F0F2FA;border:1px solid #E4E7F0;border-radius:10px;
        padding:14px 20px;margin-bottom:20px;">
      <div style="font-size:13px;font-weight:600;color:#1A1A2E;margin-bottom:6px;">
        Survey Population (Filtered): {N_filtered:,} users &nbsp;·&nbsp; How numbers are calculated</div>
      <div style="display:flex;gap:28px;flex-wrap:wrap;">
        <div style="font-size:12px;color:#6B7280;"><span style="color:#4F46E5;font-weight:600;">Opportunity</span> = users who want item but don't own it yet</div>
        <div style="font-size:12px;color:#6B7280;"><span style="color:#4F46E5;font-weight:600;">Revenue</span> = Opportunity × 20% conversion × item price</div>
        <div style="font-size:12px;color:#6B7280;"><span style="color:#4F46E5;font-weight:600;">Total Cost</span> = Product cost + Personnel cost + Campaign cost</div>
        <div style="font-size:12px;color:#6B7280;"><span style="color:#4F46E5;font-weight:600;">Gross Profit</span> = Revenue − Total Cost</div>
      </div>
    </div>""", unsafe_allow_html=True)

    k1,k2,k3,k4,k5 = st.columns(5)
    with k1: ui_card("Respondents", f"{N_filtered:,}", f"of {N} total", "👥")
    with k2: ui_card("Est. Revenue", f"₹{total_revenue:,}", "Gross Potential", "💰", gradient=True)
    with k3: ui_card("Operational Cost", f"₹{total_cost_all:,}", "SGA + CAC", "📉")
    with k4: ui_card("Gross Profit", f"₹{total_profit:,}", "Net Margin", "✅" if total_profit>0 else "⚠️")
    with k5: ui_card("ROI", f"{roi:.1f}%", f"Efficiency", "📈")

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ── AGE DISTRIBUTION ──────────────────────────────────────
    st.markdown("### 👥 Age Group Distribution")

    age_counts = filtered_survey["Age Group"].value_counts()

    fig_age = go.Figure()

    fig_age.add_trace(go.Bar(
        x=age_counts.index,
        y=age_counts.values,
        text=age_counts.values,
        textposition='outside',
        marker_color="#4F46E5"
    ))

    fig_age.update_layout(
        title="Users by Age Group",
        xaxis_title="Age Group",
        yaxis_title="Number of Users",
        height=350,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(family="Plus Jakarta Sans", color="#6B7280")
    )

    st.plotly_chart(fig_age, use_container_width=True)

    if not age_counts.empty:
        top_age = age_counts.idxmax()
        top_val = age_counts.max()

        insight_box(f"<strong>Observation:</strong> Majority users belong to <strong>{top_age}</strong> age group with <strong>{top_val}</strong> users.")

        insight_box(f"<strong>Recommendation:</strong> Target marketing campaigns specifically for <strong>{top_age}</strong> segment to maximize conversions.",
                    color="#059669", bg="#F0FDF4", border="#BBF7D0")

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ── AGE VS ACCESSORIES ────────────────────────────────────
    st.markdown("### 🎯 Age vs Accessory Preference")

    # Group by Age Group and count interests
    # We need to handle the comma separated interests correctly
    age_acc_data = []
    for _, row in filtered_survey.iterrows():
        age = row['Age Group']
        accs = str(row['Which accessories would you consider buying with a new laptop?']).split(',')
        for a in accs:
            a = a.strip()
            if a and a not in ['None','It depends','warranty','']:
                age_acc_data.append({'Age Group': age, 'Accessory': a})
    
    if age_acc_data:
        df_age_acc = pd.DataFrame(age_acc_data)
        age_acc_counts = df_age_acc.groupby(['Age Group', 'Accessory']).size().reset_index(name='Count')
        
        fig_age_acc = go.Figure()
        for age in age_filter:
            sub = age_acc_counts[age_acc_counts['Age Group'] == age]
            if not sub.empty:
                fig_age_acc.add_trace(go.Bar(
                    name=age,
                    x=sub['Accessory'],
                    y=sub['Count'],
                    text=sub['Count'],
                    textposition='outside'
                ))

        fig_age_acc.update_layout(
            title="Accessory Interest by Age Group",
            barmode='group',
            height=400,
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            font=dict(family="Plus Jakarta Sans", color="#6B7280"),
            legend=dict(orientation="h", y=-0.2)
        )

        st.plotly_chart(fig_age_acc, use_container_width=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ── CROSS-SELL ────────────────────────────────────────────
    st.markdown("""<div style="font-size:15px;font-weight:700;color:#1A1A2E;margin-bottom:16px;
        border-left:4px solid #4F46E5;padding-left:12px;">🛍️ Cross-sell — Accessory Recommendations</div>""",
        unsafe_allow_html=True)

    col_a1, col_a2 = st.columns([3, 2])
    with col_a1:
        section_label("Opportunity & Gross Profit per Accessory")
        fig1 = go.Figure()
        fig1.add_trace(go.Bar(name=f"Opportunity (of {N_filtered} users)", x=df_acc["name"], y=df_acc["opp"],
                              marker_color="#A5B4FC",
                              text=df_acc["opp"].astype(str), textposition="inside",
                              textfont=dict(size=12, color="#3730A3")))
        fig1.add_trace(go.Scatter(name="Gross Profit (₹)", x=df_acc["name"], y=df_acc["gross_profit"],
                                  yaxis="y2", mode="lines+markers",
                                  line=dict(color="#059669", width=2.5),
                                  marker=dict(size=9, color="#059669", line=dict(color="white", width=2))))
        fig1.update_layout(
            yaxis=dict(title="Users (opportunity)", gridcolor="#F0F2FA", zeroline=False, color="#9CA3AF"),
            yaxis2=dict(title="Gross Profit (₹)", overlaying="y", side="right",
                        gridcolor="#F0F2FA", zeroline=False, color="#9CA3AF", tickformat=",.0f"),
            plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
            font=dict(family="Plus Jakarta Sans", color="#6B7280", size=11),
            legend=dict(orientation="h", y=-0.30, font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=0,r=0,t=8,b=50), height=300,
            xaxis=dict(gridcolor="#F0F2FA", tickfont=dict(size=10)))
        st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar":False})

        top_acc     = df_acc.loc[df_acc["gross_profit"].idxmax()]
        top_opp_acc = df_acc.loc[df_acc["opp"].idxmax()]
        insight_box(f"<strong>Observation:</strong> <strong>{top_opp_acc['name']}</strong> has the "
                    f"largest addressable pool — <strong>{int(top_opp_acc['opp'])} out of {N_filtered} users</strong> "
                    f"want it but don't own it yet. <strong>{top_acc['name']}</strong> earns the highest "
                    f"gross profit at <strong>₹{int(top_acc['gross_profit']):,}</strong> per campaign.")
        insight_box(f"<strong>Recommendation:</strong> Bundle <strong>{top_opp_acc['name']}</strong> "
                    f"and <strong>{top_acc['name']}</strong> together. At 20% conversion these two items "
                    f"alone generate <strong>₹{int(top_opp_acc['gross_profit']+top_acc['gross_profit']):,}</strong> "
                    f"in gross profit from {N_filtered} users.",
                    color="#059669", bg="#F0FDF4", border="#BBF7D0")

    with col_a2:
        section_label("Cost Breakdown — Cross-sell")
        fig_c1 = go.Figure()
        fig_c1.add_trace(go.Bar(name="Product Cost",   x=df_acc["name"], y=df_acc["product_cost"],   marker_color="#FCA5A5"))
        fig_c1.add_trace(go.Bar(name="Personnel Cost", x=df_acc["name"], y=df_acc["personnel_cost"], marker_color="#FCD34D"))
        fig_c1.add_trace(go.Bar(name="Campaign Cost",  x=df_acc["name"], y=df_acc["campaign_cost"],  marker_color="#6EE7B7"))
        fig_c1.update_layout(barmode="stack", height=300, plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
            font=dict(family="Plus Jakarta Sans", color="#6B7280", size=10),
            legend=dict(orientation="h", y=-0.35, font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=0,r=0,t=8,b=55),
            yaxis=dict(title="Cost (₹)", gridcolor="#F0F2FA", zeroline=False, color="#9CA3AF", tickformat=",.0f"),
            xaxis=dict(gridcolor="#F0F2FA", tickfont=dict(size=9)))
        st.plotly_chart(fig_c1, use_container_width=True, config={"displayModeBar":False})
        pct_prod = int(df_acc['product_cost'].sum()/df_acc['total_cost'].sum()*100)
        pct_camp = int(df_acc['campaign_cost'].sum()/df_acc['total_cost'].sum()*100)
        insight_box(f"Product cost = <strong>{pct_prod}%</strong> of total cross-sell spend. "
                    f"Campaign ads = <strong>{pct_camp}%</strong>. "
                    f"Total cross-sell cost: <strong>₹{total_acc_cost:,}</strong>.",
                    color="#D97706", bg="#FFFBEB", border="#FDE68A")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    section_label("Cross-sell Detail Table")
    disp_acc = df_acc[["name","opp","buyers","revenue","total_cost","gross_profit"]].copy()
    disp_acc.columns = ["Accessory", f"Opportunity [{N_filtered} users]", "Est. Buyers (20%)",
                        "Revenue (₹)", "Total Cost (₹)", "Gross Profit (₹)"]
    for col in ["Revenue (₹)","Total Cost (₹)","Gross Profit (₹)"]:
        disp_acc[col] = disp_acc[col].apply(lambda v: f"₹{int(v):,}")
    st.dataframe(disp_acc, use_container_width=True, hide_index=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # ── UPSELL ────────────────────────────────────────────────
    st.markdown("""<div style="font-size:15px;font-weight:700;color:#1A1A2E;margin-bottom:16px;
        border-left:4px solid #D97706;padding-left:12px;">⚡ Upsell — Hardware Upgrade Recommendations</div>""",
        unsafe_allow_html=True)

    col_b1, col_b2 = st.columns([3, 2])
    with col_b1:
        section_label("Survey Demand & Gross Profit per Upgrade")
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="Survey Hits", x=df_up["name"], y=df_up["hits"],
                              marker_color="#C7D2FE",
                              text=df_up["hits"].astype(str)+" users", textposition="inside",
                              textfont=dict(size=11, color="#3730A3")))
        fig2.add_trace(go.Scatter(name="Gross Profit (₹)", x=df_up["name"], y=df_up["gross_profit"],
                                  yaxis="y2", mode="lines+markers",
                                  line=dict(color="#D97706", width=2.5),
                                  marker=dict(size=9, color="#D97706", line=dict(color="white", width=2))))
        fig2.update_layout(
            yaxis=dict(title="Survey Hits (users)", gridcolor="#F0F2FA", zeroline=False, color="#9CA3AF"),
            yaxis2=dict(title="Gross Profit (₹)", overlaying="y", side="right",
                        gridcolor="#F0F2FA", zeroline=False, color="#9CA3AF", tickformat=",.0f"),
            plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
            font=dict(family="Plus Jakarta Sans", color="#6B7280", size=11),
            legend=dict(orientation="h", y=-0.30, font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=0,r=0,t=8,b=50), height=300,
            xaxis=dict(gridcolor="#F0F2FA", tickfont=dict(size=10)))
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar":False})

        top_upg    = df_up.loc[df_up["gross_profit"].idxmax()]
        top_demand = df_up.loc[df_up["hits"].idxmax()]
        insight_box(f"<strong>Observation:</strong> <strong>{top_demand['name']}</strong> has the highest "
                    f"demand — <strong>{int(top_demand['hits'])} out of {N_filtered} users</strong> want this upgrade. "
                    f"<strong>{top_upg['name']}</strong> yields the best gross profit "
                    f"(<strong>₹{int(top_upg['gross_profit']):,}</strong>) despite fewer takers ({int(top_upg['hits'])} users).")
        insight_box(f"<strong>Recommendation:</strong> Use <strong>{top_demand['name']}</strong> as your "
                    f"volume driver (GP = ₹{int(top_demand['gross_profit']):,} from {int(top_demand['hits'])} users). "
                    f"Offer <strong>{top_upg['name']}</strong> separately to Premium-tier customers for "
                    f"₹{int(top_upg['gross_profit']):,} additional gross profit.",
                    color="#059669", bg="#F0FDF4", border="#BBF7D0")

    with col_b2:
        section_label("Cost Breakdown — Upsell")
        fig_c2 = go.Figure()
        fig_c2.add_trace(go.Bar(name="Product Cost",   x=df_up["name"], y=df_up["product_cost"],   marker_color="#FCA5A5"))
        fig_c2.add_trace(go.Bar(name="Personnel Cost", x=df_up["name"], y=df_up["personnel_cost"], marker_color="#FCD34D"))
        fig_c2.add_trace(go.Bar(name="Campaign Cost",  x=df_up["name"], y=df_up["campaign_cost"],  marker_color="#6EE7B7"))
        fig_c2.update_layout(barmode="stack", height=300, plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
            font=dict(family="Plus Jakarta Sans", color="#6B7280", size=10),
            legend=dict(orientation="h", y=-0.35, font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=0,r=0,t=8,b=55),
            yaxis=dict(title="Cost (₹)", gridcolor="#F0F2FA", zeroline=False, color="#9CA3AF", tickformat=",.0f"),
            xaxis=dict(gridcolor="#F0F2FA", tickfont=dict(size=9)))
        st.plotly_chart(fig_c2, use_container_width=True, config={"displayModeBar":False})
        gpu_row = df_up[df_up["name"]=="GPU Upgrade"].iloc[0]
        insight_box(f"GPU upgrade has the highest unit cost — product cost alone = "
                    f"<strong>₹{int(gpu_row['product_cost']):,}</strong> for {int(gpu_row['buyers'])} buyers. "
                    f"Total upsell cost: <strong>₹{total_upg_cost:,}</strong>.",
                    color="#D97706", bg="#FFFBEB", border="#FDE68A")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    section_label("Upsell Detail Table")
    disp_up = df_up[["name","hits","buyers","revenue","total_cost","gross_profit"]].copy()
    disp_up.columns = ["Upgrade Item", f"Survey Hits [{N_filtered} users]", "Est. Buyers (20%)",
                        "Revenue (₹)", "Total Cost (₹)", "Gross Profit (₹)"]
    for col in ["Revenue (₹)","Total Cost (₹)","Gross Profit (₹)"]:
        disp_up[col] = disp_up[col].apply(lambda v: f"₹{int(v):,}")
    st.dataframe(disp_up, use_container_width=True, hide_index=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # ── SUMMARY ───────────────────────────────────────────────
    st.markdown("""<div style="font-size:15px;font-weight:700;color:#1A1A2E;margin-bottom:16px;
        border-left:4px solid #059669;padding-left:12px;">💰 Overall Cost vs Revenue Summary</div>""",
        unsafe_allow_html=True)

    col_s1, col_s2 = st.columns([2,3])
    with col_s1:
        section_label("Revenue vs Cost — Cross-sell vs Upsell")
        cats      = ["Cross-sell","Upsell","Combined"]
        rev_vals  = [total_acc_rev,  total_upg_rev,  total_revenue]
        cost_vals = [total_acc_cost, total_upg_cost, total_cost_all]
        gp_vals   = [total_acc_rev-total_acc_cost, total_upg_rev-total_upg_cost, total_profit]
        fig_sum = go.Figure()
        fig_sum.add_trace(go.Bar(name="Revenue",      x=cats, y=rev_vals,  marker_color="#A5B4FC"))
        fig_sum.add_trace(go.Bar(name="Total Cost",   x=cats, y=cost_vals, marker_color="#FCA5A5"))
        fig_sum.add_trace(go.Bar(name="Gross Profit", x=cats, y=gp_vals,   marker_color="#6EE7B7"))
        fig_sum.update_layout(barmode="group", height=280, plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
            font=dict(family="Plus Jakarta Sans", color="#6B7280", size=11),
            legend=dict(orientation="h", y=-0.30, font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=0,r=0,t=8,b=50),
            yaxis=dict(gridcolor="#F0F2FA", zeroline=False, color="#9CA3AF", tickformat=",.0f"),
            xaxis=dict(gridcolor="#F0F2FA"))
        st.plotly_chart(fig_sum, use_container_width=True, config={"displayModeBar":False})

    with col_s2:
        section_label("Final Numbers at a Glance")
        rb = "#F0FDF4" if total_profit>0 else "#FFF7ED"
        rbo= "#BBF7D0" if total_profit>0 else "#FED7AA"
        rc = "#15803D" if total_profit>0 else "#C2410C"
        ri = "✅" if total_profit>0 else "⚠️"
        rt = "CAMPAIGN IS PROFITABLE" if total_profit>0 else f"GAP OF ₹{abs(total_profit):,} TO BREAK-EVEN"
        acc_gp = int(df_acc["gross_profit"].sum()); upg_gp = int(df_up["gross_profit"].sum())
        upg_pct = int(upg_gp/total_profit*100) if total_profit>0 else 0
        st.markdown(f"""<div style="background:{rb};border:1px solid {rbo};border-radius:12px;padding:20px 24px;">
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:14px;">
            <div>
              <div style="font-size:11px;color:#6B7280;text-transform:uppercase;letter-spacing:0.07em;font-weight:600;">Total Revenue</div>
              <div style="font-size:22px;font-weight:700;color:#1A1A2E;">₹{total_revenue:,}</div>
            </div>
            <div>
              <div style="font-size:11px;color:#6B7280;text-transform:uppercase;letter-spacing:0.07em;font-weight:600;">Total Cost</div>
              <div style="font-size:22px;font-weight:700;color:#1A1A2E;">₹{total_cost_all:,}</div>
            </div>
            <div>
              <div style="font-size:11px;color:#6B7280;text-transform:uppercase;letter-spacing:0.07em;font-weight:600;">Gross Profit</div>
              <div style="font-size:22px;font-weight:700;color:{rc};">₹{total_profit:,}</div>
            </div>
            <div>
              <div style="font-size:11px;color:#6B7280;text-transform:uppercase;letter-spacing:0.07em;font-weight:600;">ROI</div>
              <div style="font-size:22px;font-weight:700;color:{rc};">{roi:.1f}%</div>
            </div>
          </div>
          <div style="background:white;border:1px solid {rbo};border-radius:8px;padding:10px 14px;
              font-size:13px;font-weight:600;color:{rc};margin-bottom:12px;">{ri} {rt}</div>
          <div style="font-size:12px;color:#6B7280;line-height:1.8;">
            Cross-sell gross profit: <strong>₹{acc_gp:,}</strong> &nbsp;·&nbsp;
            Upsell gross profit: <strong>₹{upg_gp:,}</strong><br>
            Upsell contributes <strong>{upg_pct}%</strong> of total profit despite lower volume —
            driven by high-value hardware margins.<br>
            At 20% conversion from <strong>{N_filtered:,} survey users</strong>, the campaign
            {"generates a clear net profit." if total_profit>0 else "needs higher conversion or scale to break even."}
          </div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 3 — ASSOCIATION RULES
# ══════════════════════════════════════════════════════════════
with tab3:
    col_meta, col_table = st.columns([1,2])
    with col_meta:
        section_label("Algorithm Parameters")
        for label, val, desc, accent in [
            ("Min. Support","1%","Appears in ≥1% of listings",False),
            ("Min. Confidence","100%","Always co-bundled together",False),
            ("Min. Lift","≥ 1.0","Non-random association",False),
            ("Rules Found",str(len(rules)),"Strong cross-sell signals",True)]:
            color = "#4F46E5" if accent else "#1A1A2E"
            st.markdown(f"""<div style="background:#FFFFFF;border:1px solid #E4E7F0;border-radius:10px;
                padding:14px 16px;margin-bottom:8px;">
              <div style="font-size:11px;color:#9CA3AF;text-transform:uppercase;letter-spacing:0.06em;font-weight:600;">{label}</div>
              <div style="font-size:22px;font-weight:700;color:{color};margin:3px 0;">{val}</div>
              <div style="font-size:11px;color:#9CA3AF;">{desc}</div>
            </div>""", unsafe_allow_html=True)
        st.markdown("""<div style="background:#FFF7ED;border:1px solid #FED7AA;border-radius:10px;
            padding:12px 14px;margin-top:4px;font-size:12px;color:#92400E;line-height:1.6;">
          ⚠️ <strong>Data note:</strong> Rules derived from HP's bundled accessory catalog
          (manufacturer co-packaging), not actual customer purchase history.
        </div>""", unsafe_allow_html=True)

    with col_table:
        section_label("Discovered Rules — Confidence = 1.0, Lift ≥ 1")
        if not rules.empty:
            disp_r = rules[['antecedents','consequents','support','confidence','lift']].copy()
            disp_r['antecedents'] = disp_r['antecedents'].apply(lambda x: ', '.join(list(x)))
            disp_r['consequents'] = disp_r['consequents'].apply(lambda x: ', '.join(list(x)))
            disp_r['support']     = disp_r['support'].round(4)
            disp_r['lift']        = disp_r['lift'].round(3)
            disp_r.columns = ["If Customer Has →","Always Recommend →","Support","Confidence","Lift"]
            st.dataframe(disp_r, use_container_width=True, hide_index=True, height=280)
            top_lift_row = rules.loc[rules['lift'].idxmax()]
            top_ant = ', '.join(list(top_lift_row['antecedents']))
            top_con = ', '.join(list(top_lift_row['consequents']))
            insight_box(f"<strong>Strongest rule (Lift {top_lift_row['lift']:.2f}):</strong> Customers who have "
                        f"<em>{top_ant[:60]}</em> should always receive "
                        f"<em>{top_con[:60]}</em>. A lift of {top_lift_row['lift']:.2f} means this pairing is "
                        f"<strong>{top_lift_row['lift']:.1f}× more likely</strong> than random chance — not a coincidence.")
            st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
            section_label("Lift Score per Rule")
            rule_labels = [(", ".join(list(r['antecedents'])[:1])+" → "+", ".join(list(r['consequents'])[:1]))[:55]
                           for _, r in rules.iterrows()]
            fig3 = go.Figure(go.Bar(x=rules['lift'].values, y=rule_labels, orientation='h',
                marker_color=["#4F46E5" if v==rules['lift'].max() else "#A5B4FC" for v in rules['lift']],
                text=rules['lift'].round(2).astype(str), textposition="outside",
                textfont=dict(size=10, color="#6B7280")))
            fig3.update_layout(height=max(180,len(rules)*38), plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
                font=dict(family="Plus Jakarta Sans", color="#6B7280", size=11),
                margin=dict(l=0,r=50,t=4,b=4),
                xaxis=dict(gridcolor="#F0F2FA", title="Lift Score", zeroline=False),
                yaxis=dict(gridcolor="#F0F2FA", tickfont=dict(size=10)), showlegend=False)
            st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar":False})
        else:
            st.info("No rules found with confidence = 1.0 and lift ≥ 1.")

st.markdown("""<div style="margin-top:40px;padding-top:16px;border-top:1px solid #E4E7F0;
    display:flex;justify-content:space-between;">
  <span style="font-size:11px;color:#C0C4D0;">Smart Suggestion System </span>
  <span style="font-size:11px;color:#C0C4D0;">Apriori + Rule-based Upsell + 500-User Survey Overlay</span>
</div>""", unsafe_allow_html=True)
