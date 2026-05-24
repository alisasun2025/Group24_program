# Program title: Smart Shelf Guard - Supermarket Expiry Warning System
# Company: [Your Company Name, e.g., Yonghui Superstores / PARKnSHOP]
# Objective: Classify products by shelf-life category and detect freshness,
#            combined with business factors to generate expiry risk alerts.

# ============================================================
# Import part
# ============================================================
import streamlit as st
import numpy as np
from datetime import datetime, timedelta
from transformers import pipeline
from PIL import Image

# ============================================================
# Model loading part (cached so models load only once)
# ============================================================

# Pipeline 1: Product image → Shelf-life category classification
# ⭐ FINE-TUNED MODEL ⭐
# Fine-tuned on Grocery Store + Household Products dataset
@st.cache_resource
def load_shelf_life_classifier():
    classifier = pipeline(
        "image-classification",
        model="Alisa-Sun/shelf-life-classification"
    )
    return classifier


# Pipeline 2: Product image → Freshness detection
# ⭐ THIS IS THE FINE-TUNED MODEL ⭐
# Fine-tuned on fresh/rotten fruit dataset, comparing ViT vs ResNet vs Swin
@st.cache_resource
def load_freshness_detector():
    detector = pipeline(
        "image-classification",
        model="Alisa-Sun/freshness-detection"
    )
    return detector


# Pipeline 3: Product/shelf image → Auto-generated text description
# Pre-trained BLIP model (no fine-tuning needed)
@st.cache_resource
def load_image_captioner():
    captioner = pipeline(
        "image-text-to-text",
        model="Salesforce/blip-image-captioning-base"
    )
    return captioner


# ============================================================
# Pipeline function part
# ============================================================

def classify_shelf_life(image):
    """
    Pipeline 1: Classify product image into shelf-life category.
    Fine-tuned model directly outputs:
    - "short_shelf" (蔬果/熟食, ~1-7 days)
    - "medium_shelf" (罐头/饮料/调味品, weeks to months)
    - "non_perishable" (纸巾/洗涤用品, no expiry concern)
    """
    classifier = load_shelf_life_classifier()
    results = classifier(image, top_k=3)

    category = results[0]["label"]
    top_score = results[0]["score"]

    return category, top_score, results


def detect_freshness(image):
    """
    Pipeline 2: Detect product freshness level. ⭐ FINE-TUNED MODEL
    The fine-tuned model directly outputs:
    - "fresh" (新鲜, safe to sell)
    - "rotten" (变质, should be discarded/discounted)
    
    Model is fine-tuned on fresh/rotten fruit dataset.
    Three candidate models compared: ViT vs ResNet vs Swin.
    """
    detector = load_freshness_detector()
    results = detector(image, top_k=3)

    top_label = results[0]["label"].lower()
    top_score = results[0]["score"]

    # The fine-tuned model outputs "fresh" or "rotten" directly.
    # Map to our standard freshness categories:
    if "rotten" in top_label or "spoiled" in top_label:
        freshness = "rotten"
    elif "medium" in top_label:
        freshness = "medium_fresh"
    else:
        freshness = "fresh"

    return freshness, top_score, results


def generate_description(image):
    """
    Pipeline 3: Generate text description of the product/shelf image.
    Uses pre-trained BLIP model (no fine-tuning).
    """
    captioner = load_image_captioner()
    result = captioner(image, text="a photo of", max_new_tokens=50)
    # Handle different output formats across transformers versions
    if isinstance(result, list) and len(result) > 0:
        item = result[0]
        if isinstance(item, dict):
            description = item.get("generated_text", item.get("text", str(item)))
        elif isinstance(item, list) and len(item) > 0:
            description = item[0].get("generated_text", item[0].get("text", str(item[0])))
        else:
            description = str(item)
    else:
        description = str(result)
    return description


# ============================================================
# Business logic part (non-DL, rule-based calculations)
# ============================================================

# Shelf-life default thresholds (in days)
SHELF_LIFE_DEFAULTS = {
    "short_shelf": 7,
    "medium_shelf": 90,
    "non_perishable": 365
}

# Category display info
CATEGORY_INFO = {
    "short_shelf": {
        "label": "🥬 Short shelf-life",
        "label_cn": "Perishable goods (dairy, fruits, vegetables)",
        "color": "#e74c3c",
        "warning_ratio": 0.8
    },
    "medium_shelf": {
        "label": "🥫 Medium shelf-life",
        "label_cn": "Packaged dairy and beverages (milk, juice, yoghurt)",
        "color": "#f39c12",
        "warning_ratio": 0.8
    },
    "non_perishable": {
        "label": "🧻 Non-perishable",
        "label_cn": "Non-perishable (tissue, cleaning supplies)",
        "color": "#27ae60",
        "warning_ratio": None
    }
}

# Seasonal adjustment factors by month (1-12)
# Higher value = faster spoilage expected (e.g., summer months)
SEASONAL_FACTORS = {
    1: 0.8, 2: 0.8, 3: 0.9, 4: 1.0,
    5: 1.1, 6: 1.3, 7: 1.4, 8: 1.4,
    9: 1.2, 10: 1.0, 11: 0.9, 12: 0.8
}


def calculate_warning_threshold(shelf_life_days):
    """
    Calculate the expiry warning threshold.
    Rule: 80% of shelf life, rounded down (floor), in whole days.
    Example: 7 days → int(7 * 0.8) = int(5.6) = 5 days
    Example: 90 days → int(90 * 0.8) = int(72.0) = 72 days
    """
    return int(shelf_life_days * 0.8)


def calculate_risk_score(category, freshness, days_since_entry,
                         shelf_life_days, seasonal_factor):
    """
    Calculate overall expiry risk score (0-100, higher = more urgent).
    
    Factors:
    - Time urgency: how close to the warning threshold (0-50)
    - Freshness decay: detected freshness from Pipeline 2 (0-30)
    - Seasonal pressure: summer = higher risk (0-20)
    """
    if category == "non_perishable":
        return 0.0  # No expiry risk for non-perishable items

    # Factor 1: Time urgency (0-50 points)
    warning_days = calculate_warning_threshold(shelf_life_days)
    if days_since_entry >= shelf_life_days:
        time_score = 50  # Already expired
    elif days_since_entry >= warning_days:
        time_score = 35 + 15 * (days_since_entry - warning_days) / max(shelf_life_days - warning_days, 1)
    else:
        time_score = 35 * (days_since_entry / max(warning_days, 1))

    # Factor 2: Freshness decay (0-30 points)
    freshness_map = {"fresh": 0, "medium_fresh": 18, "rotten": 30, "N/A": 0}
    freshness_score = freshness_map.get(freshness, 0)

    # Factor 3: Seasonal pressure (0-20 points)
    seasonal_score = 20 * ((seasonal_factor - 0.8) / 0.6)
    seasonal_score = max(0, min(20, seasonal_score))

    total = time_score + freshness_score + seasonal_score
    return min(100, max(0, total))


def get_recommendation(risk_score, category):
    """Generate actionable recommendation based on risk score."""
    if category == "non_perishable":
        return "🟢 Safe", "Normal stocking, no expiry concern.", "#27ae60"

    if risk_score >= 70:
        return "🔴 Urgent", "Immediate discount or removal from shelf. Consider donation if still safe.", "#e74c3c"
    elif risk_score >= 40:
        return "🟡 Warning", "Promote sales: bundle deals, front-shelf placement, or flash discount.", "#f39c12"
    else:
        return "🟢 Safe", "Normal stocking. Monitor in next review cycle.", "#27ae60"


# ============================================================
# Main part: Streamlit user interface
# ============================================================
def main():
    st.set_page_config(
        page_title="Smart Shelf Guard",
        page_icon="🛒",
        layout="wide"
    )

    # --- Custom CSS ---
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap');

    .stApp {
        font-family: 'DM Sans', sans-serif;
    }

    .main-header {
        text-align: center;
        padding: 20px 0 10px;
    }
    .main-header h1 {
        font-size: 36px;
        font-weight: 700;
        color: #1a1a2e;
        margin: 0;
    }
    .main-header p {
        font-size: 16px;
        color: #666;
        margin-top: 4px;
    }

    .metric-card {
        background: white;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        border: 1px solid #f0f0f0;
        margin-bottom: 12px;
    }
    .metric-card h3 {
        font-size: 14px;
        color: #888;
        margin: 0 0 8px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-card .value {
        font-size: 28px;
        font-weight: 700;
    }
    .metric-card .sub {
        font-size: 13px;
        color: #999;
        margin-top: 4px;
    }

    .risk-bar {
        height: 12px;
        border-radius: 6px;
        background: #f0f0f0;
        margin-top: 10px;
        overflow: hidden;
    }
    .risk-fill {
        height: 100%;
        border-radius: 6px;
        transition: width 0.5s ease;
    }

    .rec-box {
        padding: 16px 20px;
        border-radius: 12px;
        margin-top: 12px;
        font-size: 15px;
        line-height: 1.6;
    }

    .caption-box {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 14px 18px;
        font-size: 14px;
        color: #555;
        border-left: 4px solid #667eea;
        margin-top: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

    # --- Header ---
    st.markdown("""
    <div class="main-header">
        <h1>🛒 Smart Shelf Guard</h1>
        <p>AI-powered supermarket expiry warning system — upload a product image to get started</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Sidebar: Business Parameters ---
    with st.sidebar:
        st.header("⚙️ Business Parameters")
        st.caption("Adjust these factors for risk calculation")

        st.subheader("📅 Product Info")
        entry_date = st.date_input(
            "Date entered warehouse",
            value=datetime.now() - timedelta(days=3)
        )
        shelf_life_input = st.number_input(
            "Shelf life (days)",
            min_value=1, max_value=730, value=7,
            help="Product's total shelf life in days"
        )

        st.subheader("🌡️ Seasonal Factor")
        current_month = datetime.now().month
        selected_month = st.slider(
            "Month",
            min_value=1, max_value=12,
            value=current_month,
            help="Select month to simulate seasonal impact on spoilage"
        )
        month_names = {1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr", 5:"May", 6:"Jun",
                       7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Dec"}
        seasonal_factor = SEASONAL_FACTORS[selected_month]
        st.info(
            f"📅 **{month_names[selected_month]}** → Seasonal multiplier: **{seasonal_factor}**\n\n"
            f"Seasonal risk score = 20 × ({seasonal_factor} - 0.8) / 0.6 = "
            f"**{max(0, min(20, 20 * (seasonal_factor - 0.8) / 0.6)):.1f}** / 20 points"
        )

        st.divider()
        st.subheader("📐 Threshold Rule")
        st.info(
            f"⏰ Warning threshold: **{calculate_warning_threshold(shelf_life_input)} days** "
            f"(= floor({shelf_life_input} × 80%))"
        )

    # --- Main Content Area ---
    uploaded_file = st.file_uploader(
        "📷 Upload a product image",
        type=["jpg", "jpeg", "png"],
        help="Upload a photo of the product on the shelf"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        # Layout: image left, results right
        col_img, col_result = st.columns([1, 1.5])

        with col_img:
            st.image(image, caption="Uploaded product image", use_container_width=True)

            # Pipeline 3: Image captioning
            with st.spinner("Generating image description..."):
                description = generate_description(image)
            st.markdown(
                f'<div class="caption-box">🤖 <strong>AI Description:</strong> {description}</div>',
                unsafe_allow_html=True
            )

        with col_result:
            # Pipeline 1: Shelf-life classification
            with st.spinner("Classifying product shelf-life category..."):
                category, cat_conf, cat_details = classify_shelf_life(image)

            # Pipeline 2: Freshness detection (only for short_shelf products)
            if category == "short_shelf":
                with st.spinner("Detecting product freshness..."):
                    freshness, fresh_conf, fresh_details = detect_freshness(image)
            else:
                # Non-perishable and medium_shelf: skip freshness check
                freshness = "N/A"
                fresh_conf = 0.0
                fresh_details = []

            # Calculate business metrics
            days_since_entry = (datetime.now().date() - entry_date).days
            risk_score = calculate_risk_score(
                category, freshness, days_since_entry,
                shelf_life_input, seasonal_factor
            )
            rec_label, rec_text, rec_color = get_recommendation(risk_score, category)

            # --- Display Results ---
            cat_info = CATEGORY_INFO[category]

            # Row 1: Category + Freshness
            r1c1, r1c2 = st.columns(2)
            with r1c1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Shelf-life category</h3>
                    <div class="value" style="color: {cat_info['color']}">{cat_info['label']}</div>
                    <div class="sub">{cat_info['label_cn']} · Confidence: {cat_conf:.1%}</div>
                </div>
                """, unsafe_allow_html=True)

            with r1c2:
                freshness_emoji = {"fresh": "🟢 Fresh", "medium_fresh": "🟡 Medium", "rotten": "🔴 Rotten", "N/A": "⚪ N/A"}
                freshness_colors = {"fresh": "#27ae60", "medium_fresh": "#f39c12", "rotten": "#e74c3c", "N/A": "#888"}
                freshness_sub = "Visual freshness detection · Confidence: {:.1%}".format(fresh_conf) if category == "short_shelf" else "Freshness check skipped — not a perishable product"
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Freshness level</h3>
                    <div class="value" style="color: {freshness_colors.get(freshness, '#333')}">
                        {freshness_emoji.get(freshness, freshness)}
                    </div>
                    <div class="sub">{freshness_sub}</div>
                </div>
                """, unsafe_allow_html=True)

            # Row 2: Risk Score + Recommendation
            r2c1, r2c2 = st.columns(2)
            with r2c1:
                bar_color = "#e74c3c" if risk_score >= 70 else "#f39c12" if risk_score >= 40 else "#27ae60"
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Expiry risk score</h3>
                    <div class="value" style="color: {bar_color}">{risk_score:.0f} / 100</div>
                    <div class="risk-bar">
                        <div class="risk-fill" style="width: {risk_score}%; background: {bar_color}"></div>
                    </div>
                    <div class="sub">
                        Day {days_since_entry} of {shelf_life_input} · 
                        Warning at day {calculate_warning_threshold(shelf_life_input)}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with r2c2:
                bg_map = {"🔴 Urgent": "#ffeaea", "🟡 Warning": "#fff8e1", "🟢 Safe": "#eafff0"}
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Recommendation</h3>
                    <div class="value">{rec_label}</div>
                    <div class="rec-box" style="background: {bg_map.get(rec_label, '#f5f5f5')}">
                        {rec_text}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Detailed model outputs (collapsible)
            with st.expander("🔍 View detailed model outputs"):
                st.subheader("Pipeline 1: Shelf-life classification")
                for r in cat_details:
                    st.write(f"- {r['label']}: {r['score']:.4f}")

                st.subheader("Pipeline 2: Freshness detection")
                for r in fresh_details:
                    st.write(f"- {r['label']}: {r['score']:.4f}")

    else:
        # Show placeholder when no image is uploaded
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>Pipeline 1</h3>
                <div class="value" style="font-size:20px">🏷️ Shelf-life classify</div>
                <div class="sub">ViT fine-tuned on grocery images → short / medium / non-perishable</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>Pipeline 2</h3>
                <div class="value" style="font-size:20px">🔬 Freshness detect</div>
                <div class="sub">ResNet/Swin fine-tuned on fresh/rotten dataset → fresh / medium / rotten</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>Pipeline 3</h3>
                <div class="value" style="font-size:20px">📝 Image captioning</div>
                <div class="sub">BLIP pre-trained → auto-generated product description</div>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
