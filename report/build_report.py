"""Builds the Arabic final report (docx) from the files in outputs/. Every number is read from outputs/."""
import json
import sys
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

ROOT = Path(__file__).resolve().parent.parent
T, FIG, AUD = ROOT / "outputs" / "tables", ROOT / "outputs" / "figures", ROOT / "outputs" / "audit"
FONT = "Arial"

doc = Document()
for s in doc.sections:
    s.left_margin = s.right_margin = Cm(2.2)
st = doc.styles["Normal"]
st.font.name = FONT
st.font.size = Pt(11)
st.element.rPr.rFonts.set(qn("w:cs"), FONT)


def _rtl_par(p, align=WD_ALIGN_PARAGRAPH.RIGHT):
    pPr = p._p.get_or_add_pPr()
    b = OxmlElement("w:bidi"); b.set(qn("w:val"), "1"); pPr.append(b)
    p.alignment = align
    for r in p.runs:
        rPr = r._r.get_or_add_rPr()
        rt = OxmlElement("w:rtl"); rt.set(qn("w:val"), "1"); rPr.append(rt)
        r.font.name = FONT
        rPr.rFonts.set(qn("w:cs"), FONT)


def P(text, bold=False, size=None, align=WD_ALIGN_PARAGRAPH.RIGHT):
    p = doc.add_paragraph()
    r = p.add_run(text); r.bold = bold
    if size: r.font.size = Pt(size)
    _rtl_par(p, align)
    return p


def H(text, lvl=1):
    h = doc.add_heading(text, level=lvl)
    _rtl_par(h)
    for r in h.runs:
        r.font.name = FONT
    return h


def BUL(items):
    for it in items:
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(it)
        _rtl_par(p)


def TBL(df, headers=None, fmt=None, size=9):
    fmt = fmt or {}
    df = df.copy()
    cols = list(df.columns)
    t = doc.add_table(rows=1, cols=len(cols)); t.style = "Light Grid Accent 1"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
    tblPr = t._tbl.tblPr; bv = OxmlElement("w:bidiVisual"); bv.set(qn("w:val"), "1"); tblPr.append(bv)
    for i, c in enumerate(headers or cols):
        cell = t.rows[0].cells[i]; cell.text = ""
        run = cell.paragraphs[0].add_run(str(c)); run.bold = True; run.font.size = Pt(size)
        _rtl_par(cell.paragraphs[0], WD_ALIGN_PARAGRAPH.CENTER)
    for _, row in df.iterrows():
        cells = t.add_row().cells
        for i, c in enumerate(cols):
            v = row[c]
            if c in fmt: v = fmt[c](v)
            elif isinstance(v, float): v = f"{v:,.4g}" if abs(v) < 1 else f"{v:,.2f}"
            elif isinstance(v, (int,)): v = f"{v:,}"
            cells[i].text = ""
            run = cells[i].paragraphs[0].add_run(str(v)); run.font.size = Pt(size)
            _rtl_par(cells[i].paragraphs[0], WD_ALIGN_PARAGRAPH.CENTER)
    doc.add_paragraph()


def FIGURE(name, caption, width=16):
    doc.add_picture(str(FIG / name), width=Cm(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    P(caption, size=9, align=WD_ALIGN_PARAGRAPH.CENTER)


pct = lambda x: f"{100 * x:.1f}%"
pct2 = lambda x: f"{100 * x:.2f}%"
num = lambda x: f"{int(x):,}"

# ---------------------------------------------------------------- load outputs
K = json.loads((T / "eda_key_numbers.json").read_text(encoding="utf-8"))
R = json.loads((ROOT / "outputs" / "results.json").read_text(encoding="utf-8"))
split = pd.read_csv(T / "temporal_split.csv")
funnel = pd.read_csv(T / "eligibility_funnel.csv")
comp = pd.read_csv(T / "model_comparison_test.csv")
cms = pd.read_csv(T / "confusion_matrices.csv")
impact = pd.read_csv(T / "merge_impact.csv")
mstats = pd.read_csv(T / "multi_seller_stats.csv", index_col=0)["value"]
counts = pd.read_csv(AUD / "record_counts_check.csv")
audit_b = pd.read_csv(AUD / "audit_before_tables.csv")
audit_a = pd.read_csv(AUD / "audit_after_tables.csv")
fk = pd.read_csv(AUD / "fk_orphans_after.csv")
logdf = pd.read_csv(ROOT / "outputs" / "experiments_log.csv")
dist = pd.read_csv(T / "target_distribution.csv")
coef = pd.read_csv(T / "logreg_coefficients.csv")
perm = pd.read_csv(T / "permutation_importance_selected.csv")
err_state = pd.read_csv(T / "errors_by_customer_state.csv")
err_cat = pd.read_csv(T / "errors_by_category.csv")
err_size = pd.read_csv(T / "errors_by_order_size.csv")
err_ms = pd.read_csv(T / "errors_by_multi_seller.csv")
profile = pd.read_csv(T / "error_profile_by_outcome_median.csv")
status = pd.read_csv(T / "q5_status_vs_eligible.csv")
miss = pd.read_csv(T / "q5_missing_before_after.csv")
prof_el = pd.read_csv(T / "q5_eligible_vs_excluded_profile.csv")
corr_t = pd.read_csv(T / "correlation_with_target.csv", index_col=0).iloc[:, 0]
pbin = pd.read_csv(T / "q4_late_by_planned_days_octile.csv")
q1 = pd.read_csv(T / "q1_monthly_orders_late_rate.csv")
sel_name = R["selected_model"]
M = R["models"]
sel = M[sel_name]
N, NA = K["n_eligible"], K["n_all_orders"]
base = R["baseline_test"]
test_split = split[split.split.str.startswith("test")].iloc[0]
feat_doc = (ROOT / "docs" / "feature_dictionary.md").read_text(encoding="utf-8")

# ---------------------------------------------------------------- title page
for _ in range(4): doc.add_paragraph()
P("كلية هندسة المعلوماتية - مقرر برمجة خاصة بعلم البيانات (DPD601)", bold=True, size=14, align=WD_ALIGN_PARAGRAPH.CENTER)
P("تحليل التجارة الإلكترونية والتنبؤ بتأخر تسليم الطلبات", bold=True, size=22, align=WD_ALIGN_PARAGRAPH.CENTER)
P("Brazilian E-Commerce (Olist): Late-Delivery Analysis and Prediction", size=13, align=WD_ALIGN_PARAGRAPH.CENTER)
doc.add_paragraph()
P("أعضاء الفريق (يعبّئها الفريق):", bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
for i in range(1, 6):
    P(f"الطالب {i}: ____________________   الرقم الجامعي: __________", align=WD_ALIGN_PARAGRAPH.CENTER)
doc.add_paragraph()
P("تاريخ التسليم: 2026/11/16", align=WD_ALIGN_PARAGRAPH.CENTER)
doc.add_page_break()

# ---------------------------------------------------------------- 1 intro
H("1. المقدمة")
P(f"يهدف هذا المشروع إلى تحليل بيانات الطلبات من منصة Olist البرازيلية والتنبؤ، لحظة اعتماد الطلب، بما إذا كان سيصل إلى العميل متأخراً عن "
  f"يوم التسليم المتوقع. تغطي البيانات {num(NA)} طلباً. بعد استبعاد الطلبات التي لا نتيجة تسليم لها يبقى {num(N)} طلباً مؤهلاً للتحليل الموجَّه بالهدف، "
  f"منها {num(K['n_late'])} طلباً متأخراً بنسبة {pct2(K['late_rate_overall'])}.")
P("السيناريو: يُجرى التنبؤ عند اعتماد الطلب (order_approved_at)، ونفترض أن تفاصيل العناصر والدفع وموعد التسليم المتوقع معلومة في تلك اللحظة. "
  "هذا افتراض عن لحظة إتاحة الحقول، وتعتمد عليه كل النتائج (انظر قسم القيود).")
P("المصدر والترخيص: Olist & Sionek (2018). Brazilian E-Commerce Public Dataset by Olist. Kaggle. "
  "https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce — الترخيص CC BY-NC-SA 4.0 (استخدام غير تجاري مع الإسناد). "
  "البيانات تاريخية من البرازيل (2016-2018) ولا تُعمَّم النتائج على أسواق أخرى.")
P("ملاحظة مهمة عن مصدر الملفات: لم تتوفر الحزمة الرسمية الأصلية، فاعتُمدت الملفات المنظَّفة مسبقاً من زميل (ملفات *_cleaned.csv)، "
  "والتي طُبِّق عليها استبدال القيم المفقودة في جدول المنتجات (الفئة بـ unknown والحقول الوصفية والأبعاد بالوسيط العام). "
  "وُثِّق ذلك في docs/upstream/ ولم نستطع استرجاع القيم الأصلية المفقودة؛ لذلك لا تُستخدم الحقول الوصفية (طول الاسم، طول الوصف، عدد الصور) كخصائص.")

# ---------------------------------------------------------------- 2 data
H("2. وصف البيانات")
P("الجداول السبعة المستخدمة وعدد سجلاتها (المتوقع مقابل الفعلي):")
TBL(counts, ["الجدول", "المتوقع", "الفعلي", "مطابق"], fmt={"match": lambda v: "نعم" if v else "لا"})
P("جميع الأعداد مطابقة للمتوقع، فلا اختلافات تحتاج توثيقاً في هذه النقطة.")
H("مخطط العلاقات (ERD)", 2)
P("العلاقات: طلب واحد لعدة عناصر (1:N)، وطلب واحد لعدة سجلات دفع (1:N)، وكل طلب مرتبط بسجل عميل واحد (N:1)، وعنصر الطلب مرتبط بمنتج وبائع، "
  "والمنتج مرتبط بجدول ترجمة الفئات. المخطط التفصيلي (Mermaid) في docs/schema_diagram.md:")
P("CUSTOMERS 1—N ORDERS 1—N ORDER_ITEMS N—1 PRODUCTS N—1 TRANSLATION ;  ORDER_ITEMS N—1 SELLERS ;  ORDERS 1—N ORDER_PAYMENTS", align=WD_ALIGN_PARAGRAPH.CENTER)
BUL(["customer_id: سجل عميل يُنشأ لكل طلب؛ أما customer_unique_id فهو العميل الحقيقي الذي قد يكرر الشراء. لذلك لا يصح عدّ customer_id كعدد عملاء.",
     "الطلب (order) هو عملية الشراء؛ عنصر الطلب (order item) سطر منتج واحد داخله؛ سجل الدفع (payment) هو دفعة أو طريقة دفع مستخدمة. تكرار order_id في العناصر والمدفوعات طبيعي وليس تكراراً خاطئاً."])
H("التدقيق قبل وبعد التنظيف", 2)
P("ملخص التدقيق (قبل التنظيف) لكل جدول: الصفوف، الأعمدة، الخلايا المفقودة، الصفوف المكررة بالكامل، وتفرد المفتاح. التفاصيل الكاملة بالأعمدة في outputs/audit/.")
TBL(audit_b[["table", "rows", "cols", "total_missing_cells", "full_duplicate_rows", "key", "key_unique"]],
    ["الجدول", "الصفوف", "الأعمدة", "خلايا مفقودة", "تكرار كامل", "المفتاح", "فريد"], fmt={"key_unique": lambda v: "نعم" if v else "لا"})
P(f"بعد التنظيف: مجموع الخلايا المفقودة في جدول orders = {num(audit_a.loc[audit_a.table=='orders','total_missing_cells'].iloc[0])} "
  f"(مفقودات التواريخ تُترك لأنها تعني أن الحدث لم يقع)، وفي products = {num(audit_a.loc[audit_a.table=='products','total_missing_cells'].iloc[0])} "
  "(مصدرها الأوزان الصفرية الأربعة التي حُوّلت إلى NaN). سلامة المفاتيح الأجنبية بعد التنظيف:")
TBL(fk, ["العلاقة", "صفوف الابن", "يتامى"])
P("الفئات غير الموجودة في جدول الترجمة (pc_gamer وغيرها) تُحفظ باسمها البرتغالي بدل NaN، ولذلك تظهر يتامى في تلك العلاقة فقط ومعالجتها موثقة.")

# ---------------------------------------------------------------- 3 processing
H("3. المعالجة وهندسة الخصائص")
H("التنظيف", 2)
BUL(["التواريخ حُوّلت إلى datetime بـ errors='coerce' وعدد الفشل المسجل = 0؛ الرموز البريدية بقيت نصاً من 5 خانات.",
     "لا صفوف مكررة بالكامل في أي جدول؛ لا أسعار صفرية أو سالبة ولا شحن سالب. قيم الدفع الصفرية (9) والشحن الصفري (383) صحيحة منطقياً وأُبقيت.",
     "4 منتجات بوزن صفري حُوّلت إلى NaN لتُعوَّض بوسيط التدريب فقط داخل Pipeline.",
     "ترتيب الأحداث: لا حالة اعتماد قبل الشراء؛ حالات تسليم الناقل قبل الشراء 166، وتسليم العميل قبل الناقل 23، وتسليم الناقل قبل الاعتماد 1,359. وُسمت بأعلام ولم تُحذف لأنها لا تؤثر على تعريف الهدف.",
     "كل قرار وسببه وعدد المتأثر في docs/decisions_log.md (يُولَّد آلياً)."])
H("الدمج", 2)
P("جُمّعت عناصر الطلب إلى مستوى الطلب (عدد العناصر، البائعين، المجاميع...) وجُمّعت المدفوعات على حدة، ثم دُمجت مع orders وcustomers على order_id مع validate في كل خطوة. أثر كل دمج على عدد الصفوف:")
TBL(impact, ["الخطوة", "الصفوف بعد الدمج", "order_id فريد"])
P(f"جدول النمذجة النهائي فيه صف واحد لكل طلب (تم التحقق بـ assert وباختبار آلي): {num(N)} صفاً. "
  f"طلبات متعددة البائعين: {num(mstats['multi_seller_orders'])}، متعددة الفئات: {num(mstats['multi_category_orders'])}، "
  f"وتمتد على أكثر من ولاية بائع: {num(mstats['multi_seller_state_orders'])}. القرار: ولاية البائع = ولاية البائع الرئيسي (الأكثر عناصر، ثم الأعلى قيمة، ثم seller_id) مع علم multi_seller. "
  f"الفئة الغالبة = الأكثر عناصر ثم الأعلى قيمة ثم الترتيب الأبجدي. طلبات بلا عناصر: {num(mstats['orders_without_items'])}، بلا دفع: {num(mstats['orders_without_payment'])}.")
H("تعريف الهدف والأهلية", 2)
P("is_late = 1 إذا كان يوم التسليم الفعلي للعميل بعد يوم التسليم المتوقع (مقارنة التواريخ بعد normalize لا الأوقات)، و0 إذا كان في اليوم المتوقع أو قبله. "
  "الطلبات الملغاة وغير المسلمة لا تتحول إلى فئة 0. قمع الأهلية:")
TBL(funnel[["step", "orders_remaining", "excluded_at_step"]], ["الخطوة", "المتبقي", "المستبعد في الخطوة"])
P("المدخلات الممنوعة: order_status، order_delivered_customer_date، order_delivered_carrier_date، المدة الفعلية ومقدار التأخر، و shipping_limit_date (لا تتوفر إتاحتها لحظة التنبؤ). "
  "تحقق منها اختبار آلي.")
H("الخصائص", 2)
P(f"أُنشئت {R['n_features']} خاصية (منها 4 فئوية والبقية رقمية/مشتقة) كلها متاحة لحظة الاعتماد. الجدول الكامل بطريقة الحساب والمصدر وسبب الملاءمة ووقت الإتاحة في docs/feature_dictionary.md، وملخصه:")
fd = [l.split("|")[1:-1] for l in feat_doc.splitlines() if l.startswith("| `")]
TBL(pd.DataFrame([[a.strip(" `"), b.strip(), c.strip(), e.strip()] for a, b, c, d, e in fd]),
    ["الخاصية", "طريقة الحساب", "المصدر", "وقت الإتاحة"], size=8)
P("استُبعدت الإحصاءات التاريخية لتأخر البائع/العميل لأنها تتطلب نتائج طلبات سابقة لم تكن قد سُلّمت وقت الاعتماد (خطر تسرب مرتفع). استُخدم بدلها عدد طلبات البائع السابقة (seller_prior_orders) المحسوب من الطوابع الزمنية فقط (cumcount على ترتيب زمني، وطلبات الطابع الزمني نفسه لا تُحتسب).")
H("استخدام المكتبات", 2)
BUL(["NumPy (src/numpy_ops.py): (1) أقنعة منطقية لكشف القيم غير المنطقية (price/weight <= 0) بمقارنة متجهة واحدة؛ (2) حسابات متجهة: نسبة الشحن بـ np.where لتجنب القسمة على صفر، والفرق بالأيام بـ timedelta64؛ (3) np.percentile لتحديد حدود الذيول في الرسوم؛ (4) np.log1p لتقليل التواء القيم المالية في النموذج الخطي.",
     "Pandas: قراءة بأنواع صريحة، to_datetime، .dt، groupby/agg، merge مع validate، pivot للخرائط الحرارية، qcut/cut، ودوال التحقق (is_unique)."])

# ---------------------------------------------------------------- 4 EDA
H("4. التحليل والرسوم")
P(f"المجتمع (denominator): التحليلات العامة تشمل كل الطلبات ({num(NA)})، وتحليلات الهدف تقتصر على الطلبات المؤهلة ({num(N)}). "
  f"المجموعات التي حجمها n < 100 لا تُصنَّف كأفضل/أسوأ وتُحذف أو تُظلَّل. أشرطة الخطأ هي فاصل ثقة Wilson 95%. "
  "نتائج التسليم الفعلية تُستخدم وصفياً فقط ولا تدخل النموذج. العبارات 'يظهر في البيانات' وصفية ولا تدل على سببية.")
H("س1: كيف تتغير أعداد الطلبات ونسبة التأخر عبر الزمن؟", 2)
FIGURE("fig01_monthly_orders_and_late_rate.png", "الشكل 1: الطلبات الشهرية (كل الطلبات) ونسبة التأخر (المؤهلة، أشهر n≥100)")
P(f"يظهر في البيانات نمو كبير في عدد الطلبات خلال 2017، وتذبذب واضح في نسبة التأخر: أعلى شهر شراء هو {K['peak_late_month']} بنسبة {pct(K['peak_late_rate'])} (n={num(K['peak_late_n'])})، "
  f"وأدنى شهر (n≥100) هو {K['min_late_month']} بنسبة {pct(K['min_late_rate'])}. المتوسط العام {pct(K['late_rate_overall'])}. "
  "التفسير: التأخر ليس ثابتاً بل يتركز في أشهر محددة (ذروات الطلب في نهاية 2017 وبداية 2018)، مما يعني أن توزيع الهدف يتغير مع الزمن (distribution shift) وأن التقسيم الزمني ضروري. "
  "الحد: لا تتوفر في البيانات أسباب هذه الذروات (إضرابات، ضغط لوجستي...) فلا نستطيع تفسيرها سببياً.")
P("جدول: outputs/tables/q1_monthly_orders_late_rate.csv")
H("توزيع الهدف وعدم التوازن", 2)
FIGURE("fig02_target_distribution.png", "الشكل 2: توزيع الهدف ونسبة التأخر في كل جزء زمني")
TBL(dist, ["المجموعة", "n", "متأخر", "في الوقت", "نسبة التأخر"], fmt={"late_rate": pct2})
P(f"الفئة المتأخرة أقلية ({pct2(K['late_rate_overall'])}) فالدقة وحدها مضللة: نموذج يتنبأ دائماً 'في الوقت' يحقق دقة {pct(1-K['late_rate_overall'])} دون أي فائدة. "
  f"كما تنخفض نسبة التأخر من {pct(K['train_late'])} في التدريب إلى {pct(K['val_late'])} في التحقق و{pct(K['test_late'])} في الاختبار، ما يعني أن فترة التدريب تحوي ذروات تأخر لا تتكرر في الفترة الأحدث. لذلك نعتمد Average Precision وPrecision/Recall/F1 للفئة المتأخرة.")
H("س2: هل ترتبط المسارات بين الولايات أو تعدد البائعين باختلاف التأخر؟", 2)
FIGURE("fig03_late_rate_by_customer_state.png", "الشكل 3: نسبة التأخر حسب ولاية العميل (n≥100)")
FIGURE("fig04_route_heatmap.png", "الشكل 4: خريطة حرارية للمسارات (ولاية البائع الرئيسي × ولاية العميل)")
FIGURE("fig05_multiseller_and_same_state.png", "الشكل 5: الطلبات متعددة البائعين، وتطابق ولايتي البائع والعميل", 14)
P(f"يظهر في البيانات تفاوت كبير بين الولايات: أعلى نسبة بين الولايات ذات n≥100 هي {K['worst_state']} ({pct(K['worst_state_rate'])}، n={num(K['worst_state_n'])}) وأدناها {K['best_state']} ({pct(K['best_state_rate'])}، n={num(K['best_state_n'])})؛ "
  f"و{K['n_states_small']} ولايات استُبعدت من الرسم لصغر حجمها. في الخريطة الحرارية تتجاوز مسارات SP إلى RJ وBA وES نسب 12% بينما المسارات داخل الولاية نفسها أقل (مثلاً SP→SP 4.6%). "
  f"طلبات الولاية نفسها: {pct(K['same_same_state_rate'])} (n={num(K['same_same_state_n'])}) مقابل {pct(K['same_different_state_rate'])} لولايات مختلفة (n={num(K['same_different_state_n'])}). "
  f"أما الطلبات متعددة البائعين فنسبة التأخر فيها {pct(K['multi_multi-seller_rate'])} (n={num(K['multi_multi-seller_n'])}) مقابل {pct(K['multi_single_seller_rate'])} للبائع الواحد؛ "
  "هذه النتيجة عكس التوقع، وتفسيرها المحتمل (وهو احتمال غير مثبت) أن موعد التسليم المقدَّر لهذه الطلبات أكثر تحفظاً أو أن عددها صغير ومحدود الأثر. المجموعة صغيرة نسبياً فلا تُعتمد استنتاجاً.")
H("س3: كيف تختلف نسبة التأخر باختلاف فئات المنتجات أو حجم الطلب؟", 2)
FIGURE("fig06_late_rate_by_category.png", "الشكل 6: أعلى 8 وأدنى 8 فئات من حيث نسبة التأخر (n≥100)")
FIGURE("fig07_late_rate_by_order_size_and_value.png", "الشكل 7: نسبة التأخر حسب عدد العناصر وحسب شريحة قيمة الطلب")
sr = K["size_rates"]
P(f"من {K['n_categories_total']} فئة، تحقق {K['n_categories_ge_min']} فئة فقط شرط n≥100 وهي وحدها المصنَّفة. أعلى نسبة بينها {K['cat_worst']} ({pct(K['cat_worst_rate'])}، n={num(K['cat_worst_n'])}) وأدناها {K['cat_best']} ({pct(K['cat_best_rate'])}، n={num(K['cat_best_n'])}). "
  f"الفروق الصغيرة بين الفئات تحتاج حذراً لأن أشرطة الخطأ في الشكل 6 عريضة للفئات ذات n القريب من 100. بالنسبة لحجم الطلب: عنصر واحد {pct(sr['1 item'][0])} (n={num(sr['1 item'][1])})، عنصران {pct(sr['2 items'][0])}، ثلاثة {pct(sr['3 items'][0])}، 4 فأكثر {pct(sr['4+ items'][0])}. "
  "لا يظهر أن الطلبات الأكبر أكثر تأخراً؛ التفسير الممكن أن الطلبات ذات العنصر الواحد هي الأغلبية الساحقة وتضم كل أنواع المنتجات والمسارات.")
H("س4: ما العلاقة بين المدة المخططة للتسليم والتأخر الفعلي؟", 2)
FIGURE("fig08_planned_days_by_outcome.png", "الشكل 8: توزيع المدة المخططة حسب النتيجة (هيستوغرام وصندوقي)")
FIGURE("fig09_planned_vs_actual.png", "الشكل 9: المدة المخططة مقابل الفعلية (نقاط) ونسبة التأخر حسب شرائح المدة المخططة")
oct_ = pbin
P(f"الوسيط المخطط للطلبات في الوقت {K['planned_median_ontime']:.1f} يوماً وللمتأخرة {K['planned_median_late']:.1f} يوماً؛ أي أن المدة المخططة وحدها لا تفصل الفئتين. "
  f"الارتباط (Spearman) بين المدة المخططة والفعلية {K['corr_planned_actual']:.2f}: الوعد يتبع المدة الفعلية جزئياً. "
  f"نسبة التأخر ليست خطية مع المدة المخططة: من {pct(oct_.late_rate.iloc[0])} في أقصر شريحة إلى {pct(oct_.late_rate.max())} كحد أعلى وإلى {pct(oct_.late_rate.iloc[-1])} في أطول شريحة. "
  "أي أن التأخر يظهر عند الطلبات ذات الوعد المتوسط والقصير ويقل بوضوح عند الوعود الطويلة جداً (هامش أمان)، لكن العلاقة ضعيفة بما يكفي لتبرير نماذج غير خطية. لا نستنتج سببية.")
H("العلاقات بين المتغيرات", 2)
FIGURE("fig10_correlation_heatmap.png", "الشكل 10: مصفوفة ارتباط Spearman بين الخصائص الرقمية والهدف", 14)
top = corr_t.head(3)
P("أعلى ارتباطات الخصائص بالهدف (Spearman): " + "، ".join(f"{k} ({v:+.3f})" for k, v in top.items()) + ". "
  "كلها ارتباطات ضعيفة (|ρ| < 0.1)، ويتوافق هذا مع ضعف القدرة التنبؤية لاحقاً. ارتباط الخصائص المالية بعضها ببعض أعلى (السعر والدفع والشحن) مما يعني تكرار معلومات.")
H("س5: ما أثر القيم المفقودة والاستبعادات على تمثيل البيانات؟", 2)
FIGURE("fig11_exclusions_impact.png", "الشكل 11: قمع الأهلية، حالات الطلب، وحصة الولايات قبل/بعد الاستبعاد", 17)
TBL(status, ["حالة الطلب", "كل الطلبات", "المؤهل"])
TBL(prof_el.assign(stat=prof_el.stat.map({"median": "وسيط", "mean": "متوسط"})), ["الخاصية", "المقياس", "المؤهلة", "المستبعدة"])
P(f"استُبعد {num(K['n_excluded'])} طلباً ({pct(K['n_excluded']/NA)}) معظمها طلبات غير مسلمة (ملغاة، غير متاحة، قيد الشحن). الخصائص الوصفية (السعر، المدة المخططة، حجم الطلب) متقاربة بين المؤهلة والمستبعدة، أي أن الاستبعاد لا يشوّه التمثيل بشكل ظاهر على هذه الأبعاد، "
  "لكن نسبة الطلبات متعددة البائعين في المستبعدة أقل بكثير (انظر الجدول). المفقودات المرتبطة بتواريخ التسليم تعني أن النتيجة غير معروفة، ولذلك لا تُعتبر 'في الوقت'. "
  "تحذير مهم: الطلبات التي لم تُسلَّم بعد نهاية فترة البيانات أو تأخرت كثيراً قد تكون ناقصة التمثيل في الأشهر الأخيرة (censoring)، مما قد يُنقص نسبة التأخر المرصودة فيها.")
H("س6: ما الأنماط التي تستحق المتابعة وما حدود تفسيرها؟", 2)
BUL([f"ذروة التأخر في {K['peak_late_month']} ({pct(K['peak_late_rate'])}) وتفاوت الولايات (من {pct(K['best_state_rate'])} إلى {pct(K['worst_state_rate'])})، وهما يستحقان متابعة تشغيلية، خصوصاً مسارات SP→RJ/BA/ES.",
     f"الطلبات داخل الولاية الواحدة أقل تأخراً ({pct(K['same_same_state_rate'])} مقابل {pct(K['same_different_state_rate'])}).",
     "المدة المخططة الطويلة جداً مرتبطة بتأخر أقل؛ ويستحق الأمر دراسة كيف يُحسب الموعد المقدَّر.",
     "الحدود: بيانات ملاحظة (ليست تجربة) فلا سببية؛ مجموعات صغيرة (n<100) لم تُصنَّف؛ فترة 21 شهراً من سوق واحد؛ اختلاف توزيع الهدف بين الفترات؛ مفقودات منتجات عُوّضت قبل استلامنا للبيانات."])

# ---------------------------------------------------------------- 5 modelling
H("5. النماذج والتقييم")
P("التقسيم زمني حسب order_approved_at (بدون خلط عشوائي): 70% تدريب، 15% تحقق، 15% اختبار نهائي (الأحدث). الحدود وأعداد الأجزاء:")
TBL(split, ["الجزء", "البداية", "النهاية", "n", "متأخر", "نسبة التأخر"], fmt={"late_rate": pct2, "n_late": num})
P("كل المحولات (imputer بالوسيط، log1p، StandardScaler، OneHotEncoder) داخل sklearn Pipeline وتُضبط على التدريب فقط. "
  "النماذج: DummyClassifier(prior) كمرجع، وقاعدة 'كل الطلبات متأخرة'، ثم Logistic Regression (class_weight=balanced) وHistGradientBoosting وRandom Forest. "
  "الإعدادات اختيرت بشبكة بحث على التحقق (Average Precision)، والعتبة اختيرت لتعظيم F1 على التحقق، ثم طُبِّقت مرة واحدة على الاختبار. لم نُعد تدريب النماذج على التدريب+التحقق حفاظاً على صلاحية العتبة.")
n_tune = int((logdf.stage == "tuning").sum())
P(f"عدد تجارب الضبط المسجلة: {n_tune} (في outputs/experiments_log.csv مع الإعدادات والخصائص والمقاييس والبذرة 42). أفضل الإعدادات: " +
  "؛ ".join(f"{k}: {v['params']}" for k, v in M.items()) + ".")
H("جدول المقارنة على الاختبار النهائي (الفئة المتأخرة = 1)", 2)
TBL(comp, ["النموذج", "Accuracy", "Precision", "Recall", "F1", "Average Precision", "AP على التحقق"], size=8)
FIGURE("fig_pr_curves.png", "الشكل 12: منحنيات Precision-Recall على التحقق والاختبار")
H("مصفوفات الارتباك (الاختبار)", 2)
FIGURE("fig_confusion_matrices_test.png", "الشكل 13: مصفوفات الارتباك على الاختبار بالعتبة المختارة على التحقق")
TBL(cms[cms.split == "test"], ["النموذج", "الجزء", "TN", "FP", "FN", "TP", "العتبة"])
hgb = M["HistGradientBoosting"]; lr = M["Logistic Regression"]; rf = M["Random Forest"]
P(f"النتيجة بصدق: النماذج الثلاثة تتفوق على المرجع في Average Precision (المرجع {base['average_precision']:.3f} يساوي نسبة التأخر في الاختبار) لكن التفوق متواضع. "
  f"النموذج المختار بحسب التحقق فقط هو {sel_name} (AP على التحقق {sel['val']['average_precision']:.3f}، وعلى الاختبار {sel['test']['average_precision']:.3f}، Precision {sel['test']['precision']:.3f}، Recall {sel['test']['recall']:.3f}، F1 {sel['test']['f1']:.3f}). "
  f"على الاختبار حقق HistGradientBoosting AP={hgb['test']['average_precision']:.3f} وF1={hgb['test']['f1']:.3f} وهما أعلى من النموذج المختار، لكن لم نُغيّر الاختيار بعد رؤية الاختبار حفاظاً على نزاهة التقييم؛ نذكر هذا الفارق كملاحظة. "
  f"قاعدة 'الكل متأخر' تعطي F1={R['always_late_test']['f1']:.3f} وRecall=1، فتفوق النماذج عليها في F1 محدود (Logistic {lr['test']['f1']:.3f}، HGB {hgb['test']['f1']:.3f}، Random Forest {rf['test']['f1']:.3f}) وRandom Forest أقل منها، وPrecision بين {rf['test']['precision']:.2f} و{hgb['test']['precision']:.2f} فقط: معظم الإنذارات خاطئة. "
  "الاستنتاج: الخصائص المتاحة لحظة الاعتماد تحمل إشارة ضعيفة عن التأخر ولا تكفي لنظام إنذار دقيق.")
H("أهم الخصائص", 2)
FIGURE("fig_feature_importance.png", "الشكل 14: أهمية الخصائص (Permutation على التحقق) ومعاملات الانحدار اللوجستي")
P("أهم خصائص النموذج المختار بحسب permutation importance على التحقق: " + "، ".join(f"{r.feature} ({r.importance_mean:.3f})" for r in perm.head(5).itertuples()) +
  ". أكبر المعاملات في الانحدار اللوجستي: " + "، ".join(f"{r.feature.replace('cat__','').replace('num_log__','')} ({r.coefficient:+.2f})" for r in coef.head(4).itertuples()) +
  ". تحذير: الأهمية تعكس ارتباطاً بالتنبؤ وليس سبباً للتأخر، والخصائص مترابطة (السعر والشحن والدفع) فتتوزع الأهمية بينها، وولاية العميل تمثل جزئياً فترات زمنية ومسارات لوجستية.")
H("تحليل الأخطاء (الاختبار، النموذج المختار)", 2)
es = err_state[err_state.reliable_n_ge_100].head(6)
TBL(es[["customer_state", "n", "late_rate", "predicted_late_rate", "precision", "recall"]], ["ولاية العميل", "n", "نسبة التأخر الفعلية", "نسبة المتنبأ بتأخرها", "Precision", "Recall"],
    fmt={"late_rate": pct, "predicted_late_rate": pct, "precision": lambda v: "-" if pd.isna(v) else f"{v:.2f}", "recall": lambda v: "-" if pd.isna(v) else f"{v:.2f}"})
rj = err_state[err_state.customer_state == "RJ"].iloc[0] if (err_state.customer_state == "RJ").any() else None
sp = err_state[err_state.customer_state == "SP"].iloc[0]
msg = (f"في ولاية RJ (n={int(rj.n)}) يتنبأ النموذج بتأخر {pct(rj.predicted_late_rate)} من الطلبات بينما التأخر الفعلي {pct(rj.late_rate)}، فأغلب الإيجابيات الكاذبة (FP={int(rj.fp)}) تأتي منها؛ سبب محتمل أن التدريب شمل فترات تأخر مرتفع جداً لمسارات RJ (ذروة 2018-03) ولم تتكرر في الاختبار. "
       if rj is not None else "")
P(msg + f"وفي SP (n={int(sp.n)}) تأخر فعلي {pct(sp.late_rate)} لكن Recall {sp.recall:.2f} فقط (FN={int(sp.fn)})، أي أن أكبر عدد من السلبيات الكاذبة (متأخرة لم تُكتشف) في أكبر ولاية. الأرقام الكاملة: outputs/tables/errors_by_*.csv.")
TBL(err_size[["order_size", "n", "late_rate", "predicted_late_rate", "precision", "recall"]], ["حجم الطلب", "n", "نسبة التأخر", "نسبة المتنبأ", "Precision", "Recall"],
    fmt={"late_rate": pct, "predicted_late_rate": pct, "precision": lambda v: "-" if pd.isna(v) else f"{v:.2f}", "recall": lambda v: "-" if pd.isna(v) else f"{v:.2f}"})
pf = profile.set_index("feature")
P(f"الوسيط لخصائص أنماط الأخطاء: المدة المخططة للسلبيات الكاذبة (FN) {pf.loc['planned_days','FN']:.1f} يوماً مقابل {pf.loc['planned_days','FP']:.1f} للإيجابيات الكاذبة (FP) و{pf.loc['planned_days','TP']:.1f} للإيجابيات الصحيحة (TP): "
  "يفوّت النموذج الطلبات المتأخرة ذات الوعد القصير نسبياً والسعر المنخفض ويتوقع التأخر لطلبات أغلى ومدتها المخططة أطول. هذه أنماط وصفية على عينة الاختبار.")

# ---------------------------------------------------------------- 6 conclusions
H("6. الاستنتاجات")
BUL([f"نسبة التأخر العامة {pct2(K['late_rate_overall'])} ({num(K['n_late'])} من {num(N)} طلباً مؤهلاً)، وهي غير ثابتة زمنياً (من {pct(K['min_late_rate'])} إلى {pct(K['peak_late_rate'])} حسب الشهر).",
     "يظهر في البيانات أن مسار الشحن (ولاية العميل، تطابق الولايتين) والزمن أكثر ارتباطاً بالتأخر من حجم الطلب أو عدد عناصره.",
     f"أفضل نموذج بحسب التحقق: {sel_name} بـ AP={sel['test']['average_precision']:.3f} على الاختبار مقابل {base['average_precision']:.3f} للمرجع؛ التفوق حقيقي لكنه متواضع ولا يكفي لنظام تشغيلي دقيق.",
     "القدرة التنبؤية المحدودة تعكس غياب معلومات مهمة لحظة الاعتماد (حالة الناقل، الأحوال التشغيلية)، وتغيّر توزيع التأخر عبر الزمن."])

# ---------------------------------------------------------------- 7 limitations
H("7. القيود")
BUL(["لحظة إتاحة الحقول: افترضنا أن العناصر والدفع وموعد التسليم المقدَّر معلومة عند الاعتماد؛ إن لم يكن كذلك تتغير الخصائص المتاحة.",
     "الملفات المدخلة منظَّفة مسبقاً (تعويض الوسيط العام للحقول الوصفية والأبعاد) وليست الحزمة الأصلية؛ نسبة صغيرة من القيم قد تكون معوَّضة بمعلومة من مجمل البيانات.",
     "تغير توزيع التأخر عبر الزمن (distribution shift): فترة التدريب تحوي ذروات غير موجودة في الاختبار، فتتأثر العتبة والـ Precision.",
     "احتمال censoring في الأشهر الأخيرة (طلبات لم تُسلَّم بعد).",
     "الاستبعاد: لم تُستخدم الإحصاءات التاريخية للبائع بسبب خطر التسرب.",
     "عدم تعميم: بيانات سوق واحد وفترة 2016-2018، ولا تدل الارتباطات على سببية.",
     "العتبة ضُبطت على التحقق ثم لم يُعاد تدريب النماذج على التدريب+التحقق، فقد يكون الأداء أقل من الأمثل."])

# ---------------------------------------------------------------- 8 refs
H("8. المراجع")
BUL(["Olist & Sionek, A. (2018). Brazilian E-Commerce Public Dataset by Olist. Kaggle. https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce (CC BY-NC-SA 4.0).",
     "Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. JMLR 12.",
     "McKinney, W. (2010). Data Structures for Statistical Computing in Python. Harris et al. (2020). Array programming with NumPy. Nature 585.",
     "Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. JASA 22."])

# ---------------------------------------------------------------- 9 contributions
H("9. المساهمات")
TBL(pd.DataFrame([["جودة البيانات والتنظيف", "", "", "feature/data-quality"], ["الدمج والخصائص", "", "", "feature/merge-features"],
                  ["التحليل والرسوم", "", "", "feature/eda"], ["النماذج والتقييم", "", "", "feature/modeling"],
                  ["الاختبارات والتوثيق", "", "", "feature/tests-docs"]]), ["العمل", "الصاحب", "المراجع", "الفرع"])
P("(تُعبّأ الأسماء من قبل الفريق؛ انظر docs/CONTRIBUTIONS.md وdocs/REVIEW_CHECKLIST.md)")

out = ROOT / "report" / "DPD601_Final_Report_AR.docx"
doc.save(out)
print("saved", out)
