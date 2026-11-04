"""Molecular Virtual Laboratory v2 (Streamlit + RDKit)"""
import os
import re
from datetime import date
from io import BytesIO
from urllib.parse import quote
from xml.sax.saxutils import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import py3Dmol
import requests
import streamlit as st
import streamlit.components.v1 as components
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw, QED, rdMolDescriptors, rdMolTransforms
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image as RLImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

st.set_page_config(page_title="Molecular Virtual Lab", page_icon="🧪", layout="wide")
st.markdown("<style>.main-title{text-align:center;font-size:40px;font-weight:bold;}</style>",
            unsafe_allow_html=True)
BASE = os.path.dirname(os.path.abspath(__file__))
LOGO = os.path.join(BASE, "srm_logo.jpeg")
DEVELOPER, DEVELOPER_INFO = "Dharun Gopi J", "B.Tech CSE (Cybersecurity), Section Cyber B"
PT = Chem.GetPeriodicTable()

# ---------------- suggestions ----------------
SUGGEST = {
    "Water (H2O)": "O", "Carbon dioxide (CO2)": "O=C=O", "Ammonia (NH3)": "N",
    "Methane (CH4)": "C", "Oxygen (O2)": "O=O", "Nitrogen (N2)": "N#N",
    "Hydrogen (H2)": "[H][H]", "Ozone (O3)": "[O-][O+]=O", "Hydrogen peroxide (H2O2)": "OO",
    "Sulfuric acid (H2SO4)": "OS(=O)(=O)O", "Ethanol (C2H5OH)": "CCO",
    "Acetic acid (CH3COOH)": "CC(=O)O", "Benzene (C6H6)": "c1ccccc1",
    "Glucose (C6H12O6)": "OCC1OC(O)C(O)C(O)C1O", "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C", "Paracetamol": "CC(=O)Nc1ccc(O)cc1",
}
LOOKUP = {}
for _label, _smi in SUGGEST.items():
    for _part in re.split(r"\s*[()]\s*", _label):
        if _part:
            LOOKUP[_part.lower()] = (_smi, _label)
ELEM_FORMS = {"H": "[H][H]", "N": "N#N", "O": "O=O", "F": "FF", "Cl": "ClCl", "Br": "BrBr", "I": "II"}

GROUPS = {
    "Alcohol (-OH)": "[CX4][OX2H]", "Carboxylic acid (-COOH)": "C(=O)[OX2H1]",
    "Aldehyde": "[CX3H1](=O)[#6]", "Ketone": "[#6][CX3](=O)[#6]", "Ester": "[#6][CX3](=O)[OX2][#6]",
    "Amine": "[NX3;!$(NC=O)]", "Amide": "C(=O)[NX3]", "Ether": "[OD2]([#6])[#6]",
    "Aromatic ring": "a1aaaaa1", "Halide": "[F,Cl,Br,I]", "Alkene (C=C)": "C=C",
    "Alkyne (C#C)": "C#C", "Nitro": "[N+](=O)[O-]", "Thiol (-SH)": "[SX2H]",
    "Sulfonyl / S=O": "S=O", "Phosphorus group": "[P]",
}
QUIZ = [
    ("What is the molecular formula of water?", ["HO₂", "H₂O₂", "H₂O"], "H₂O"),
    ("Which molecule contains a triple bond?", ["O₂", "N₂", "Cl₂"], "N₂"),
    ("How many valence electrons does oxygen have?", ["4", "8", "6"], "6"),
    ("How many π bonds are present in O=O?", ["0", "1", "2"], "1"),
    ("Which of these is a compound (not an element)?", ["O₃", "CO₂", "Cl₂"], "CO₂"),
    ("Which molecule has a bent shape?", ["CO₂", "CH₄", "H₂O"], "H₂O"),
    ("A high TPSA value suggests the molecule is...", ["Very polar", "Non-polar", "Radioactive"], "Very polar"),
    ("In Lipinski's rule, the molecular weight limit is...", ["200", "500", "900"], "500"),
    ("A positive, high LogP means the molecule is...", ["Water-loving", "Fat-loving (lipophilic)", "Acidic"], "Fat-loving (lipophilic)"),
    ("Which functional group is -COOH?", ["Alcohol", "Amine", "Carboxylic acid"], "Carboxylic acid"),
]


# ---------------- molecule lookup ----------------
@st.cache_data(show_spinner=False)
def pubchem(name):
    try:
        url = (f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{quote(name)}"
               "/property/SMILES,IUPACName/JSON")
        p = requests.get(url, timeout=10).json()["PropertyTable"]["Properties"][0]
        smi = p.get("SMILES") or p.get("IsomericSMILES") or p.get("ConnectivitySMILES")
        return smi, p.get("IUPACName"), p.get("CID")
    except Exception:
        return None, None, None


@st.cache_data(show_spinner=False, ttl=86400)
def pubchem_ids(smiles):
    """Return (CID, IUPAC name) from PubChem for a SMILES string, or (None, None)."""
    try:
        r = requests.post(
            "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/property/IUPACName/JSON",
            data={"smiles": smiles},
            timeout=10,
        )
        p = r.json()["PropertyTable"]["Properties"][0]
        return p.get("CID"), p.get("IUPACName")
    except Exception:
        return None, None


def resolve(q):
    """Return (smiles, title, iupac, cid) for a typed name / formula / element / SMILES."""
    if q.lower() in LOOKUP:
        smi, label = LOOKUP[q.lower()]
        return smi, label, None, None
    try:
        PT.GetAtomicNumber(q)  # exact element symbol such as Na, Fe, Cl
        return ELEM_FORMS.get(q, f"[{q}]"), f"Element {q}", None, None
    except Exception:
        pass
    if Chem.MolFromSmiles(q) is not None:
        return q, q, None, None
    smi, iupac, cid = pubchem(q)
    if smi and Chem.MolFromSmiles(smi) is not None:
        return smi, q.title(), iupac, cid
    return None, None, None, None


# ---------------- chemistry helpers ----------------
def embed3d(mol):
    m = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(m, randomSeed=42) != 0 and \
            AllChem.EmbedMolecule(m, randomSeed=42, useRandomCoords=True) != 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(m)
    except Exception:
        pass
    return m


def png_2d(mol, size=(450, 350)):
    buf = BytesIO()
    Draw.MolToImage(mol, size=size).save(buf, format="PNG")
    return buf.getvalue()


CPK = {"H": "#d9d9d9", "C": "#404040", "N": "#3050f8", "O": "#e8000d", "S": "#e6c200",
       "P": "#ff8000", "Cl": "#1ff01f", "F": "#90e050", "Br": "#a62929", "I": "#940094"}


def png_3d(m3):
    """Static ball-and-stick picture of the 3D model (used inside the PDF)."""
    pts = m3.GetConformer().GetPositions()
    fig = plt.figure(figsize=(4.5, 3.8))
    ax = fig.add_subplot(111, projection="3d")
    for b in m3.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        ax.plot(*zip(pts[i], pts[j]), color="gray", lw=2.5)
    for a in m3.GetAtoms():
        x, y, z = pts[a.GetIdx()]
        ax.scatter(x, y, z, s=70 if a.GetSymbol() == "H" else 260,
                   color=CPK.get(a.GetSymbol(), "#c040c0"), edgecolors="k", depthshade=False)
    c, r = pts.mean(axis=0), max(abs(pts - pts.mean(axis=0)).max(), 1.0)
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(20, 35)
    ax.set_axis_off()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def show_3d(m3, style):
    v = py3Dmol.view(width=520, height=400)
    v.addModel(Chem.MolToMolBlock(m3), "mol")
    v.setStyle({"stick": {"radius": 0.15}, "sphere": {"scale": 0.3}} if style == "Ball & Stick"
               else {"stick": {}} if style == "Stick" else {"sphere": {}})
    v.setBackgroundColor("white")
    v.zoomTo()
    components.html(v._make_html(), height=410)


def bond_counts(mol):
    names = {1.0: "Single", 2.0: "Double", 3.0: "Triple", 1.5: "Aromatic"}
    c = {"Single": 0, "Double": 0, "Triple": 0, "Aromatic": 0}
    for b in Chem.AddHs(mol).GetBonds():
        c[names.get(b.GetBondTypeAsDouble(), "Single")] += 1
    return c


PROPS = {
    "Molecular formula": rdMolDescriptors.CalcMolFormula,
    "Molecular weight (g/mol)": Descriptors.MolWt,
    "Exact mass": Descriptors.ExactMolWt,
    "Heavy atoms": lambda m: m.GetNumHeavyAtoms(),
    "Total atoms (with H)": lambda m: Chem.AddHs(m).GetNumAtoms(),
    "Total bonds (sigma bonds)": lambda m: Chem.AddHs(m).GetNumBonds(),
    "Pi bonds": lambda m: (lambda c: int(c["Double"] + 2 * c["Triple"] + c["Aromatic"] / 2))(bond_counts(m)),
    "Valence electrons": Descriptors.NumValenceElectrons,
    "Formal charge": Chem.GetFormalCharge,
    "LogP (Crippen)": Descriptors.MolLogP,
    "Molar refractivity": Descriptors.MolMR,
    "TPSA (Å²)": rdMolDescriptors.CalcTPSA,
    "H-bond donors": rdMolDescriptors.CalcNumHBD,
    "H-bond acceptors": rdMolDescriptors.CalcNumHBA,
    "Rotatable bonds": rdMolDescriptors.CalcNumRotatableBonds,
    "Rings": rdMolDescriptors.CalcNumRings,
    "Aromatic rings": rdMolDescriptors.CalcNumAromaticRings,
    "Heteroatoms": rdMolDescriptors.CalcNumHeteroatoms,
    "Fraction Csp3": rdMolDescriptors.CalcFractionCSP3,
    "Stereocentres": lambda m: len(Chem.FindMolChiralCenters(m, includeUnassigned=True)),
    "Radical electrons": Descriptors.NumRadicalElectrons,
    "Complexity (BertzCT)": Descriptors.BertzCT,
    "QED drug-likeness": QED.qed,
}


def properties(mol):
    out = {}
    for k, f in PROPS.items():
        try:
            v = f(mol)
        except Exception:
            v = "n/a"
        out[k] = round(v, 3) if isinstance(v, float) else v
    return out


def num(P, k):
    return P[k] if isinstance(P[k], (int, float)) else 0.0


def identity(mol, iupac, cid):
    d = {"Canonical SMILES": Chem.MolToSmiles(mol), "IUPAC name": iupac or "not available (offline)",
         "PubChem CID": cid or "not available"}
    for k, f in (("InChI", Chem.MolToInchi), ("InChIKey", Chem.MolToInchiKey)):
        try:
            d[k] = f(mol)
        except Exception:
            d[k] = "n/a"
    return d


def composition(mol):
    counts = {}
    for a in Chem.AddHs(mol).GetAtoms():
        counts[a.GetSymbol()] = counts.get(a.GetSymbol(), 0) + 1
    tot = sum(PT.GetAtomicWeight(s) * n for s, n in counts.items())
    return pd.DataFrame([{
        "Element": s, "Atomic no.": PT.GetAtomicNumber(s),
        "Atomic mass (u)": round(PT.GetAtomicWeight(s), 3), "Valence e-": PT.GetNOuterElecs(s),
        "Covalent radius (Å)": round(PT.GetRcovalent(s), 2), "Atoms": n,
        "Mass %": round(100 * PT.GetAtomicWeight(s) * n / tot, 2)} for s, n in counts.items()])


def functional_groups(mol):
    out = {}
    for name, sma in GROUPS.items():
        n = len(mol.GetSubstructMatches(Chem.MolFromSmarts(sma)))
        if n:
            out[name] = n
    return out


def lipinski(P):
    rules = {"Molecular weight <= 500": num(P, "Molecular weight (g/mol)") <= 500,
             "LogP <= 5": num(P, "LogP (Crippen)") <= 5,
             "H-bond donors <= 5": num(P, "H-bond donors") <= 5,
             "H-bond acceptors <= 10": num(P, "H-bond acceptors") <= 10}
    return rules


def atom_table(mol):
    mh = Chem.AddHs(mol)
    try:
        AllChem.ComputeGasteigerCharges(mh)
    except Exception:
        pass
    rows = []
    for a in mh.GetAtoms():
        try:
            q = round(float(a.GetProp("_GasteigerCharge")), 3)
        except Exception:
            q = float("nan")
        rows.append({"Atom": f"{a.GetSymbol()}{a.GetIdx()}", "Hybridization": str(a.GetHybridization()),
                     "Neighbours": a.GetDegree(), "Formal charge": a.GetFormalCharge(),
                     "Gasteiger charge": q, "Aromatic": a.GetIsAromatic()})
    return pd.DataFrame(rows)


def bond_table(m3):
    conf = m3.GetConformer()
    rows = []
    for b in m3.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        rows.append({"Bond": f"{m3.GetAtomWithIdx(i).GetSymbol()}{i}-{m3.GetAtomWithIdx(j).GetSymbol()}{j}",
                     "Type": str(b.GetBondType()),
                     "Length (Å)": round(rdMolTransforms.GetBondLength(conf, i, j), 3)})
    return pd.DataFrame(rows)


# ---------------- PDF report ----------------
@st.cache_data(show_spinner=False)
def build_pdf(smiles, title, iupac, cid, student, reg):
    mol = Chem.MolFromSmiles(smiles)
    m3 = embed3d(mol)
    P = properties(mol)
    S = getSampleStyleSheet()
    N = S["Normal"]

    def grid(data, widths):
        rows = [[Paragraph(escape(str(c)), N) for c in r] for r in data]
        t = Table(rows, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                               ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey)]))
        return t

    story = []
    if os.path.exists(LOGO):
        story.append(RLImage(LOGO, width=3.2 * inch, height=1.6 * inch))
    story += [Paragraph("Molecular Properties Report", S["Title"]),
              Paragraph(f"<b>Molecule:</b> {escape(title)}<br/><b>Student:</b> {escape(student or 'Not provided')}"
                        f" &nbsp; <b>Reg. No.:</b> {escape(reg or 'Not provided')}"
                        f" &nbsp; <b>Date:</b> {date.today():%d-%m-%Y}", N), Spacer(1, 8)]

    story.append(Paragraph("1. Molecular structures", S["Heading2"]))
    pic2 = RLImage(BytesIO(png_2d(mol)), width=3 * inch, height=2.33 * inch)
    cells = [[pic2, RLImage(BytesIO(png_3d(m3)), width=3 * inch, height=2.53 * inch) if m3 else "3D model not available"],
             ["2D structure", "3D structure (ball and stick)"]]
    story += [Table(cells, colWidths=[3.1 * inch, 3.1 * inch]), Spacer(1, 8)]

    sections = [
        ("2. Identification", [["Item", "Value"]] + [[k, v] for k, v in identity(mol, iupac, cid).items()], [1.5, 4.7]),
        ("3. Physicochemical properties", [["Property", "Value"]] + [[k, v] for k, v in P.items()], [3, 3.2]),
    ]
    comp = composition(mol)
    sections.append(("4. Elemental composition", [list(comp.columns)] + comp.values.tolist(), [0.9, 0.8, 0.9, 0.8, 1.0, 0.7, 0.7]))
    fg = functional_groups(mol)
    sections.append(("5. Functional groups", [["Group", "Count"]] + ([[k, v] for k, v in fg.items()] or [["None detected", ""]]), [3, 3.2]))
    sections.append(("6. Lipinski rule of five", [["Rule", "Passed?"]] + [[k, "Yes" if v else "No"] for k, v in lipinski(P).items()], [3, 3.2]))
    if m3:
        bt = bond_table(m3).head(30)
        sections.append(("7. Bond lengths (3D model)", [list(bt.columns)] + bt.values.tolist(), [2, 2.1, 2.1]))
    for head, data, w in sections:
        story += [Paragraph(head, S["Heading2"]), grid(data, [x * inch for x in w]), Spacer(1, 6)]
    buf = BytesIO()
    SimpleDocTemplate(buf, pagesize=A4, topMargin=30, bottomMargin=30).build(story)
    return buf.getvalue()


# ---------------- sidebar ----------------
st.sidebar.title("🧪 Virtual Lab")
page = st.sidebar.radio("Navigation", [
    "🏠 Home", "📖 Theory", "🧬 Molecular Visualization", "📊 Molecular Properties Report",
    "📈 Structure–Property Analysis", "📝 Quiz"])
st.sidebar.divider()
st.sidebar.markdown("### 👩‍🎓 Student Information")
student = st.sidebar.text_input("Student Name", key="student_name")
reg = st.sidebar.text_input("Registration Number", key="registration_number")
st.sidebar.divider()
st.sidebar.markdown("### 🔎 Choose a molecule")
pick = st.sidebar.selectbox("Suggested molecules", list(SUGGEST))
typed = st.sidebar.text_input("Or type your own", placeholder="name, formula, element or SMILES",
                              help="Examples: aspirin, NaCl, H2SO4, Fe, CCO, c1ccccc1")

smiles = title = iupac = cid = mol = m3 = None
if typed.strip():
    smiles, title, iupac, cid = resolve(typed.strip())
    if smiles is None:
        st.sidebar.error("Could not find that molecule. Check the spelling or type its SMILES.")
else:
    smiles, title = SUGGEST[pick], pick
if smiles:
    if iupac is None or cid is None:
        cid, iupac = pubchem_ids(smiles)
    mol = Chem.MolFromSmiles(smiles)
    m3 = embed3d(mol)
    st.sidebar.download_button("📄 Download full PDF report", build_pdf(smiles, title, iupac, cid, student, reg),
                               file_name="molecular_report.pdf", mime="application/pdf")

MOL_PAGES = ["🧬 Molecular Visualization", "📊 Molecular Properties Report", "📈 Structure–Property Analysis"]
if page in MOL_PAGES:
    if mol is None:
        st.warning("Pick or type a valid molecule in the sidebar.")
        st.stop()
    P = properties(mol)
    st.caption(f"Selected molecule: **{title}** | SMILES: `{smiles}`")

# ---------------- pages ----------------
if page == "🏠 Home":
    if os.path.exists(LOGO):
        _, mid, _ = st.columns([1, 4, 1])
        mid.image(LOGO)
    st.markdown('<div class="main-title">🧪 Molecular Virtual Laboratory</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align:center;font-size:22px;font-weight:700;color:#1f4e79;">'
                'SRM Institute of Science and Technology, Tiruchirappalli</div>'
                '<div style="text-align:center;font-size:17px;font-weight:600;">'
                'School of Sciences | Division of Chemistry</div>', unsafe_allow_html=True)
    st.divider()
    st.markdown("""
## Welcome
Type **any** molecule, compound or element in the sidebar (or pick a suggestion) and explore it:

🧬 **Molecular Visualization**: 2D and interactive 3D structures

📊 **Molecular Properties Report**: identifiers, 20+ properties, composition, atoms, bonds, functional groups

📈 **Structure–Property Analysis**: Lipinski rule, polarity and comparison with other molecules

📄 **PDF report** with both 2D and 3D structures, downloadable from the sidebar

📝 **Quiz** to test yourself
""")
    st.info("You can type a name (aspirin), a formula (H2SO4, NaCl), an element (Fe) or a SMILES string (CCO). "
            "Names and formulas that are not in the suggestion list are searched on PubChem, so internet is needed.")
    st.markdown(f"### 👨‍🎓 Developed by\n**{DEVELOPER}**  \n{DEVELOPER_INFO}  \nSRM Institute of Science and Technology, Tiruchirappalli")

elif page == "📖 Theory":
    st.title("📖 Theory")
    st.markdown("""
**Molecular formula** gives the number of atoms of each element; **structural formula** shows how they are bonded.

**SMILES** is a text code for a structure (`O` = water, `CCO` = ethanol).

**Molecular weight** is the sum of the atomic masses. **LogP** measures fat- versus water-loving behaviour.

**TPSA** is the polar surface area; high TPSA means a more polar molecule.

**H-bond donors/acceptors** decide how strongly a molecule bonds with water.

**Lipinski's rule of five**: MW ≤ 500, LogP ≤ 5, donors ≤ 5, acceptors ≤ 10 for good oral drug-likeness.

**σ and π bonds**: every bond has one σ bond; a double bond adds one π bond and a triple bond adds two.
""")

elif page == "🧬 Molecular Visualization":
    st.title("🧬 Molecular Visualization")
    left, right = st.columns(2)
    with left:
        st.markdown("**2D structure**")
        png = png_2d(mol)
        st.image(png)
        st.download_button("⬇️ 2D structure (PNG)", png, file_name="structure_2d.png", mime="image/png")
    with right:
        st.markdown("**3D structure** (drag to rotate, scroll to zoom)")
        if m3:
            show_3d(m3, st.selectbox("3D style", ["Ball & Stick", "Stick", "Space Filling"]))
            st.download_button("⬇️ 3D model (MOL file)", Chem.MolToMolBlock(m3), file_name="structure_3d.mol")
        else:
            st.warning("A 3D model could not be generated for this molecule.")
    c = st.columns(4)
    c[0].metric("Formula", P["Molecular formula"])
    c[1].metric("Mol. weight", f"{P['Molecular weight (g/mol)']} g/mol")
    c[2].metric("Atoms (with H)", P["Total atoms (with H)"])
    c[3].metric("Bonds", P["Total bonds (sigma bonds)"])

elif page == "📊 Molecular Properties Report":
    st.title("📊 Molecular Properties Report")
    t1, t2, t3, t4, t5 = st.tabs(["🆔 Identity", "⚗️ Properties", "🧮 Composition", "⚛️ Atoms & Bonds", "🧩 Functional groups"])
    with t1:
        for k, v in identity(mol, iupac, cid).items():
            st.markdown(f"**{k}:**")
            st.code(str(v), language=None)
    with t2:
        df = pd.DataFrame({"Property": list(P), "Value": [str(v) for v in P.values()]})
        st.dataframe(df, hide_index=True)
        st.download_button("⬇️ Properties (CSV)", df.to_csv(index=False).encode("utf-8-sig"),
                           file_name="properties.csv", mime="text/csv")
    with t3:
        st.dataframe(composition(mol), hide_index=True)
    with t4:
        st.markdown("**Bond types**")
        st.dataframe(pd.DataFrame([bond_counts(mol)]), hide_index=True)
        st.markdown("**Atom details (hybridization and partial charges)**")
        st.dataframe(atom_table(mol), hide_index=True)
        if m3:
            st.markdown("**Bond lengths from the 3D model**")
            st.dataframe(bond_table(m3), hide_index=True)
    with t5:
        fg = functional_groups(mol)
        if fg:
            st.dataframe(pd.DataFrame({"Functional group": list(fg), "Count": list(fg.values())}), hide_index=True)
        else:
            st.info("No common functional groups detected.")

elif page == "📈 Structure–Property Analysis":
    st.title("📈 Structure–Property Analysis")
    st.subheader("1. Lipinski rule of five")
    rules = lipinski(P)
    st.dataframe(pd.DataFrame({"Rule": list(rules), "Passed": ["✅" if v else "❌" for v in rules.values()]}), hide_index=True)
    st.write("**Verdict:** " + ("Drug-like (no violations)." if all(rules.values())
                                else f"{sum(not v for v in rules.values())} violation(s), so poor oral drug-likeness is likely."))
    st.subheader("2. What the structure tells us")
    lp, tp = num(P, "LogP (Crippen)"), num(P, "TPSA (Å²)")
    st.write(f"- **Lipophilicity:** LogP = {lp}, so the molecule is " +
             ("fat-loving (lipophilic)." if lp > 3 else "water-loving (hydrophilic)." if lp < 1 else "moderately lipophilic."))
    st.write(f"- **Polarity:** TPSA = {tp} Å², so it is " +
             ("highly polar." if tp > 90 else "moderately polar." if tp > 30 else "mostly non-polar."))
    st.write(f"- **Hydrogen bonding:** {P['H-bond donors']} donor(s) and {P['H-bond acceptors']} acceptor(s). "
             "More donors/acceptors usually give higher boiling point and water solubility.")
    st.write(f"- **Flexibility:** {P['Rotatable bonds']} rotatable bond(s) and {P['Rings']} ring(s).")
    fg = functional_groups(mol)
    st.write("- **Functional groups:** " + (", ".join(f"{k} ×{v}" for k, v in fg.items()) if fg else "none detected."))
    st.subheader("3. Compare with other molecules")
    picks = st.multiselect("Compare with", list(SUGGEST), default=list(SUGGEST)[:3])
    rows = []
    for label, smi in [(title, smiles)] + [(p, SUGGEST[p]) for p in picks]:
        q = properties(Chem.MolFromSmiles(smi))
        rows.append({"Molecule": label, "MW": num(q, "Molecular weight (g/mol)"), "LogP": num(q, "LogP (Crippen)"),
                     "TPSA": num(q, "TPSA (Å²)"), "HBD": num(q, "H-bond donors"), "HBA": num(q, "H-bond acceptors")})
    cmp_df = pd.DataFrame(rows)
    st.dataframe(cmp_df, hide_index=True)
    st.bar_chart(cmp_df.set_index("Molecule")[["LogP", "TPSA"]])
    st.bar_chart(cmp_df.set_index("Molecule")[["MW"]])

elif page == "📝 Quiz":
    st.title("📝 Quiz")
    answers = [st.radio(f"Q{i}. {q}", o, index=None, key=f"quiz_{i}") for i, (q, o, _) in enumerate(QUIZ, 1)]
    if st.button("Submit Quiz"):
        if None in answers:
            st.warning("Please answer all the questions before submitting.")
        else:
            score = sum(a == k for a, (_, _, k) in zip(answers, QUIZ))
            pct = score / len(QUIZ) * 100
            st.success(f"{student or 'Student'}, your score: {score}/{len(QUIZ)} ({pct:.0f}%)")
            if pct >= 80:
                st.balloons()
            with st.expander("Review answers"):
                for i, (a, (q, _, k)) in enumerate(zip(answers, QUIZ), 1):
                    st.write(f"{'✅' if a == k else '❌'} Q{i}. {q} — correct answer: **{k}**")

st.divider()
st.caption("🧪 Molecular Virtual Laboratory | Python, Streamlit, RDKit and PubChem")
