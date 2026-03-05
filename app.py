import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.colors import LinearSegmentedColormap
from io import BytesIO
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="HS Baseball Cards", page_icon="⚾", layout="wide")

# ── Style ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Mono&family=DM+Sans:wght@400;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; background:#0d0d0d; color:#e8e8e8; }
.stApp { background:#0d0d0d; }
</style>
""", unsafe_allow_html=True)

# ── Colors ────────────────────────────────────────────────────────────────────
BG_COLOR     = '#0d0d0d'
PANEL_COLOR  = '#161616'
BORDER_COLOR = '#2a2a2a'
TEXT_COLOR   = '#e8e8e8'
DIM_COLOR    = '#888888'

# ── Stat definitions ──────────────────────────────────────────────────────────
# (column, display label, higher=better, format)
STAT_DEFS = [
    ('avg',  'AVG',  True,  '.3f'),
    ('obp',  'OBP',  True,  '.3f'),
    ('slg',  'SLG',  True,  '.3f'),
    ('ops',  'OPS',  True,  '.3f'),
    ('hr',   'HR',   True,  'd'),
    ('rbi',  'RBI',  True,  'd'),
    ('r',    'R',    True,  'd'),
    ('h',    'H',    True,  'd'),
    ('2b',   '2B',   True,  'd'),
    ('3b',   '3B',   True,  'd'),
    ('sb',   'SB',   True,  'd'),
    ('bb',   'BB',   True,  'd'),
    ('k',    'K',    False, 'd'),
    ('kpct', 'K%',   False, '.1f'),
    ('bbpct','BB%',  True,  '.1f'),
]

# Stats shown in the big header box
HEADER_STATS = ['avg','obp','slg','ops','hr','rbi','sb']

# Stats shown in percentile bar section
PCT_STATS = ['avg','obp','slg','ops','hr','rbi','r','h','2b','3b','sb','bb','k','kpct','bbpct']

# ── Compute derived stats ─────────────────────────────────────────────────────
def compute_stats(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d.columns = [c.strip().lower() for c in d.columns]

    # Fill missing optional columns with 0
    for col in ['hbp','sf','2b','3b','hr','bb','k','sb','r','rbi']:
        if col not in d.columns:
            d[col] = 0
    d = d.fillna(0)

    # Numeric coerce
    for col in ['ab','h','2b','3b','hr','r','rbi','bb','k','sb','hbp','sf']:
        d[col] = pd.to_numeric(d[col], errors='coerce').fillna(0).astype(int)

    # Singles
    d['1b'] = d['h'] - d['2b'] - d['3b'] - d['hr']

    # AVG
    d['avg'] = np.where(d['ab'] > 0, d['h'] / d['ab'], 0)

    # OBP = (H + BB + HBP) / (AB + BB + HBP + SF)
    obp_num = d['h'] + d['bb'] + d['hbp']
    obp_den = d['ab'] + d['bb'] + d['hbp'] + d['sf']
    d['obp'] = np.where(obp_den > 0, obp_num / obp_den, 0)

    # SLG = (1B + 2*2B + 3*3B + 4*HR) / AB
    slg_num = d['1b'] + 2*d['2b'] + 3*d['3b'] + 4*d['hr']
    d['slg'] = np.where(d['ab'] > 0, slg_num / d['ab'], 0)

    # OPS
    d['ops'] = d['obp'] + d['slg']

    # K% and BB%
    pa = d['ab'] + d['bb'] + d['hbp'] + d['sf']
    d['kpct']  = np.where(pa > 0, d['k']  / pa * 100, 0)
    d['bbpct'] = np.where(pa > 0, d['bb'] / pa * 100, 0)
    d['pa'] = pa

    return d

# ── Multi-year aggregation ────────────────────────────────────────────────────
def aggregate_player(df: pd.DataFrame, name: str) -> pd.Series:
    """Sum counting stats across years, then recompute rates."""
    p = df[df['player_norm'] == name.strip().lower()].copy()
    if p.empty:
        return None

    agg = {}
    agg['player'] = p['player'].iloc[0]
    agg['team']   = p['team'].iloc[-1]   # most recent team
    agg['pos']    = p['pos'].iloc[-1]
    agg['years']  = sorted(p['year'].astype(int).tolist())
    agg['year_str'] = ' / '.join(str(y) for y in agg['years'])

    for col in ['ab','h','2b','3b','hr','r','rbi','bb','k','sb','hbp','sf']:
        agg[col] = int(p[col].sum())

    # Recompute rates from career totals
    tmp = pd.DataFrame([agg])
    tmp = compute_stats(tmp)
    row = tmp.iloc[0]

    for stat in ['avg','obp','slg','ops','kpct','bbpct','pa','1b']:
        agg[stat] = row[stat]

    return pd.Series(agg)

# ── Percentile calculation ────────────────────────────────────────────────────
def get_percentile(value, series, higher_is_better=True):
    """Return 0–100 percentile rank."""
    s = series.dropna()
    if len(s) == 0: return 50
    if higher_is_better:
        return float(np.sum(s <= value) / len(s) * 100)
    else:
        return float(np.sum(s >= value) / len(s) * 100)

# ── Percentile color (Savant style: blue=bad → white=avg → red=good) ─────────
def pct_color(pct):
    if pct >= 50:
        t = (pct - 50) / 50
        r = int(255 * t + 220 * (1-t))
        g = int(50  * t + 220 * (1-t))
        b = int(50  * t + 220 * (1-t))
    else:
        t = pct / 50
        r = int(220 * t + 40  * (1-t))
        g = int(220 * t + 80  * (1-t))
        b = int(220 * t + 255 * (1-t))
    return f"#{r:02x}{g:02x}{b:02x}"

# ── Build the matplotlib card ─────────────────────────────────────────────────
def build_card(player_series: pd.Series, league_df: pd.DataFrame, min_ab: int) -> BytesIO:
    p = player_series

    # Qualified league pool (career AB >= min_ab)
    qual = league_df[league_df['ab'] >= min_ab].copy()

    # ── Figure layout ─────────────────────────────────────────────────────────
    FIG_W = 20
    fig = plt.figure(figsize=(FIG_W, 14), facecolor=BG_COLOR)
    outer = gridspec.GridSpec(3, 1, figure=fig,
        height_ratios=[2.8, 5.5, 5.5],
        hspace=0.18, left=0.04, right=0.96, top=0.97, bottom=0.03)

    # ── HEADER ────────────────────────────────────────────────────────────────
    ax_hdr = fig.add_subplot(outer[0])
    ax_hdr.set_facecolor(PANEL_COLOR); ax_hdr.axis('off')
    ax_hdr.set_xlim(0,1); ax_hdr.set_ylim(0,1)

    # Red accent bar
    ax_hdr.plot([0,1],[0.997,0.997], color='#cc0000', linewidth=4,
                transform=ax_hdr.transAxes, clip_on=False)

    # Name + info
    ax_hdr.text(0.01, 0.72, p['player'].upper(),
                transform=ax_hdr.transAxes, color=TEXT_COLOR,
                fontsize=22, fontweight='bold', va='center')
    subtitle = f"{p['team']}  ·  {p['pos']}  ·  {p['year_str']}"
    ax_hdr.text(0.01, 0.25, subtitle,
                transform=ax_hdr.transAxes, color=DIM_COLOR, fontsize=11, va='center')

    # Qualified badge
    is_qual = p['ab'] >= min_ab
    badge_color = '#2ecc71' if is_qual else '#e74c3c'
    badge_text  = f"QUALIFIED  ({int(p['ab'])} AB)" if is_qual else f"NOT QUALIFIED  ({int(p['ab'])} AB / {min_ab} req)"
    ax_hdr.text(0.01, 0.0, badge_text, transform=ax_hdr.transAxes,
                color=badge_color, fontsize=9, fontweight='bold', va='bottom')

    # Stat boxes
    stat_labels = {
        'avg':'AVG','obp':'OBP','slg':'SLG','ops':'OPS',
        'hr':'HR','rbi':'RBI','sb':'SB','ab':'AB'
    }
    keys = list(stat_labels.keys())
    n_boxes = len(keys)
    box_x0 = 0.30; box_w = 0.68; box_h = 0.88; box_y0 = 0.06
    cell_w = box_w / n_boxes

    ax_hdr.add_patch(FancyBboxPatch((box_x0, box_y0), box_w, box_h,
        boxstyle='round,pad=0.005', facecolor='#1a1a1a',
        edgecolor=BORDER_COLOR, linewidth=1.2,
        transform=ax_hdr.transAxes, zorder=2))

    for i, key in enumerate(keys):
        cx = box_x0 + cell_w * (i + 0.5)
        val = p.get(key, 0)
        # Format value
        if key in ['avg','obp','slg','ops']:
            val_str = f"{float(val):.3f}".lstrip('0') if float(val) < 1 else f"{float(val):.3f}"
        else:
            val_str = str(int(val))

        # Percentile color for value text
        stat_row = next((s for s in STAT_DEFS if s[0]==key), None)
        if stat_row and key in qual.columns and len(qual) > 0:
            pct = get_percentile(float(val), qual[key], stat_row[2])
            col = pct_color(pct)
        else:
            col = TEXT_COLOR

        ax_hdr.text(cx, box_y0 + box_h*0.68, val_str,
                    transform=ax_hdr.transAxes, color=col,
                    fontsize=15, fontweight='bold', ha='center', va='center', zorder=3)
        ax_hdr.text(cx, box_y0 + box_h*0.22, stat_labels[key],
                    transform=ax_hdr.transAxes, color='#cccccc',
                    fontsize=10, ha='center', va='center', zorder=3)
        if i < n_boxes - 1:
            div_x = box_x0 + cell_w * (i+1)
            ax_hdr.plot([div_x,div_x],[box_y0+0.06, box_y0+box_h-0.06],
                        color=BORDER_COLOR, linewidth=0.7,
                        transform=ax_hdr.transAxes, zorder=3)

    # ── PERCENTILE BARS (top half) ────────────────────────────────────────────
    pct_stats_top = ['avg','obp','slg','ops','hr','rbi','r','h']
    pct_stats_bot = ['2b','3b','sb','bb','k','kpct','bbpct']

    def draw_pct_section(ax, stat_keys):
        ax.set_facecolor(PANEL_COLOR); ax.axis('off')
        ax.set_xlim(0,1); ax.set_ylim(0,1)

        n = len(stat_keys)
        row_h = 0.92 / n
        y_start = 0.97

        for i, key in enumerate(stat_keys):
            stat_row = next((s for s in STAT_DEFS if s[0]==key), None)
            if stat_row is None: continue
            _, label, higher_better, fmt = stat_row

            val = float(p.get(key, 0))
            pct = get_percentile(val, qual[key], higher_better) if key in qual.columns and len(qual)>0 else 50
            color = pct_color(pct)

            y = y_start - row_h * (i + 0.5)
            bar_left  = 0.18
            bar_right = 0.78
            bar_width = bar_right - bar_left
            bar_h     = row_h * 0.32
            bar_y     = y - bar_h / 2

            # Stat label
            ax.text(bar_left - 0.02, y, label,
                    ha='right', va='center', color=DIM_COLOR,
                    fontsize=11, fontweight='bold', transform=ax.transAxes)

            # Background track
            ax.add_patch(FancyBboxPatch((bar_left, bar_y), bar_width, bar_h,
                boxstyle='round,pad=0.002', facecolor='#252525',
                edgecolor='none', transform=ax.transAxes, zorder=1))

            # Fill bar
            fill_w = max(bar_width * pct / 100, 0.004)
            ax.add_patch(FancyBboxPatch((bar_left, bar_y), fill_w, bar_h,
                boxstyle='round,pad=0.002', facecolor=color,
                edgecolor='none', transform=ax.transAxes, zorder=2))

            # 50th percentile tick
            mid_x = bar_left + bar_width * 0.5
            ax.plot([mid_x, mid_x], [bar_y - 0.005, bar_y + bar_h + 0.005],
                    color='#555555', linewidth=1.0,
                    transform=ax.transAxes, zorder=3)

            # Value text
            if fmt == '.3f':
                val_str = f"{val:.3f}".lstrip('0') if val < 1 else f"{val:.3f}"
            elif fmt == '.1f':
                val_str = f"{val:.1f}%"
            else:
                val_str = str(int(val))

            ax.text(bar_right + 0.02, y, val_str,
                    ha='left', va='center', color=color,
                    fontsize=11, fontweight='bold', transform=ax.transAxes)

            # Percentile label
            ax.text(0.99, y, f"{int(pct)}th",
                    ha='right', va='center', color=DIM_COLOR,
                    fontsize=9, transform=ax.transAxes)

        # Section title
        ax.text(0.01, 0.99, 'PERCENTILE RANKINGS  (vs qualified league hitters)',
                ha='left', va='top', color=TEXT_COLOR,
                fontsize=10, fontweight='bold', transform=ax.transAxes)

        # Legend
        for pct_val, lbl in [(10,'10th'),(25,'25th'),(50,'50th'),(75,'75th'),(90,'90th')]:
            lx = bar_left + bar_width * pct_val/100
            ax.text(lx, 0.01, lbl, ha='center', va='bottom',
                    color='#444444', fontsize=7.5, transform=ax.transAxes)

    # Split percentile bars across two panels
    body_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1], wspace=0.12)
    ax_pct_l = fig.add_subplot(body_gs[0])
    ax_pct_r = fig.add_subplot(body_gs[1])
    draw_pct_section(ax_pct_l, pct_stats_top)
    draw_pct_section(ax_pct_r, pct_stats_bot)

    # ── COUNTING STATS TABLE ──────────────────────────────────────────────────
    ax_tbl = fig.add_subplot(outer[2])
    ax_tbl.set_facecolor(PANEL_COLOR); ax_tbl.axis('off')
    ax_tbl.set_xlim(0,1); ax_tbl.set_ylim(0,1)

    tbl_cols = ['ab','h','1b','2b','3b','hr','r','rbi','bb','k','sb','hbp','pa']
    tbl_labels = ['AB','H','1B','2B','3B','HR','R','RBI','BB','K','SB','HBP','PA']

    n_cols = len(tbl_cols)
    cell_w2 = 1.0 / n_cols

    # Header
    for ci, lbl in enumerate(tbl_labels):
        ax_tbl.text((ci+0.5)*cell_w2, 0.88, lbl,
                    ha='center', va='center', color='#cccccc',
                    fontsize=12, fontweight='bold', transform=ax_tbl.transAxes)

    ax_tbl.plot([0.01, 0.99], [0.72, 0.72],
                color=BORDER_COLOR, linewidth=0.8, transform=ax_tbl.transAxes)

    # Values (career totals)
    for ci, key in enumerate(tbl_cols):
        val = p.get(key, 0)
        val_str = str(int(float(val)))

        stat_row = next((s for s in STAT_DEFS if s[0]==key), None)
        if stat_row and key in qual.columns and len(qual)>0:
            pct = get_percentile(float(val), qual[key], stat_row[2])
            col = pct_color(pct)
        else:
            col = TEXT_COLOR

        ax_tbl.text((ci+0.5)*cell_w2, 0.42, val_str,
                    ha='center', va='center', color=col,
                    fontsize=14, fontweight='bold', transform=ax_tbl.transAxes)

    # League averages row
    ax_tbl.plot([0.01, 0.99], [0.28, 0.28],
                color=BORDER_COLOR, linewidth=0.5, transform=ax_tbl.transAxes)
    ax_tbl.text(0.01, 0.18, 'LG AVG', ha='left', va='center',
                color=DIM_COLOR, fontsize=9, transform=ax_tbl.transAxes)
    for ci, key in enumerate(tbl_cols):
        if key in qual.columns:
            avg_val = qual[key].mean()
            if key in ['avg','obp','slg','ops']:
                avg_str = f"{avg_val:.3f}".lstrip('0') if avg_val < 1 else f"{avg_val:.3f}"
            else:
                avg_str = f"{avg_val:.1f}"
            ax_tbl.text((ci+0.5)*cell_w2, 0.12, avg_str,
                        ha='center', va='center', color=DIM_COLOR,
                        fontsize=9, transform=ax_tbl.transAxes)

    ax_tbl.text(0.01, 0.97, 'CAREER COUNTING STATS',
                ha='left', va='top', color=TEXT_COLOR,
                fontsize=11, fontweight='bold', transform=ax_tbl.transAxes)
    ax_tbl.text(0.99, 0.97, f"League pool: {len(qual)} qualified players (min {min_ab} AB)",
                ha='right', va='top', color=DIM_COLOR,
                fontsize=9, transform=ax_tbl.transAxes)

    # Border on all panels
    for ax in [ax_hdr, ax_pct_l, ax_pct_r, ax_tbl]:
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER_COLOR)

    fig.text(0.5, 0.005, 'Stats via MaxPreps  ·  Percentiles vs qualified league hitters',
             ha='center', va='bottom', color='#555555', fontsize=9, style='italic')

    buf = BytesIO()
    plt.savefig(buf, dpi=180, bbox_inches='tight',
                facecolor=BG_COLOR, edgecolor='none', format='png')
    plt.close(fig)
    buf.seek(0)
    return buf

# ── Streamlit UI ──────────────────────────────────────────────────────────────
st.title("⚾ High School Baseball Player Cards")

# File upload
uploaded = st.file_uploader(
    "Upload your league stats CSV",
    type="csv",
    help="See the template below for the required format"
)

with st.expander("📋 CSV Format / Template"):
    st.markdown("""
Your CSV needs these columns (column names are flexible — just keep them consistent):

| player | team | year | pos | ab | h | 2b | 3b | hr | r | rbi | bb | k | sb | hbp | sf |
|--------|------|------|-----|----|---|----|----|----|----|-----|----|---|----|-----|-----|
| John Smith | Lincoln High | 2024 | SS | 87 | 31 | 6 | 1 | 3 | 22 | 18 | 12 | 14 | 8 | 2 | 1 |

- **hbp** (hit by pitch) and **sf** (sac fly) are optional but improve OBP accuracy
- Add multiple rows per player for multiple seasons — the app combines them automatically
- One CSV covers your whole league across all years
    """)
    # Provide template download
    template = pd.DataFrame([{
        'player':'John Smith','team':'Lincoln High','year':2024,'pos':'SS',
        'ab':87,'h':31,'2b':6,'3b':1,'hr':3,'r':22,'rbi':18,'bb':12,'k':14,'sb':8,'hbp':2,'sf':1
    }])
    st.download_button("⬇ Download blank template",
                       data=template.to_csv(index=False),
                       file_name="league_stats_template.csv",
                       mime="text/csv")

if uploaded:
    try:
        raw = pd.read_csv(uploaded)
        raw.columns = [c.strip().lower() for c in raw.columns]
        df = compute_stats(raw)
        df['player_norm'] = df['player'].str.strip().str.lower()

        # Aggregate each player across years
        all_players = df['player_norm'].unique()
        career_rows = [aggregate_player(df, name) for name in all_players]
        career_df = pd.DataFrame(career_rows).reset_index(drop=True)

        st.success(f"Loaded {len(df)} season rows · {len(career_df)} unique players")

        # Min AB slider
        suggested_min = int(career_df['ab'].quantile(0.40))
        min_ab = st.slider(
            "Minimum AB to qualify for percentiles",
            min_value=10, max_value=int(career_df['ab'].max()),
            value=suggested_min, step=5
        )
        n_qual = (career_df['ab'] >= min_ab).sum()
        st.caption(f"{n_qual} of {len(career_df)} players qualify at {min_ab} AB threshold")

        # Player search
        st.markdown("---")
        player_names = sorted(career_df['player'].tolist())
        search = st.selectbox("Search player", options=[""] + player_names)

        if search:
            player_row = career_df[career_df['player']==search].iloc[0]
            with st.spinner("Building card..."):
                buf = build_card(player_row, career_df, min_ab)
            st.image(buf, use_column_width=True)
            st.download_button(
                label="⬇ Download Card PNG",
                data=buf,
                file_name=f"{search.replace(' ','_')}_card.png",
                mime="image/png"
            )

    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        st.exception(e)
else:
    st.info("Upload your league CSV above to get started.")
