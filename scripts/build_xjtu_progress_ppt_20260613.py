from __future__ import annotations

import html
import os
import posixpath
import shutil
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "xjtu_template.pptx"
OUT = ROOT / "GeoNexus_RSD_progress_xjtu_20260613.pptx"
BUILD_ROOT = ROOT / "artifacts"
ASSETS = ROOT / "artifacts" / "ppt_assets_20260608"

P_NS = "http://schemas.openxmlformats.org/presentationml/2006/main"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
CT_NS = "http://schemas.openxmlformats.org/package/2006/content-types"

ET.register_namespace("p", P_NS)
ET.register_namespace("a", A_NS)
ET.register_namespace("r", R_NS)


EMU_W = 12_192_000
EMU_H = 6_858_000


def emu(v: float) -> int:
    return int(v * 914400)


def esc(text: str) -> str:
    return html.escape(text, quote=True)


def run_text(text: str, size: int = 22, color: str = "1F2937", bold: bool = False) -> str:
    b = ' b="1"' if bold else ""
    return (
        f'<a:r><a:rPr lang="en-US" sz="{size * 100}"{b}>'
        f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>'
        f'<a:latin typeface="Aptos"/></a:rPr><a:t>{esc(text)}</a:t></a:r>'
    )


def paragraph(text: str, size: int = 22, color: str = "1F2937", bold: bool = False) -> str:
    return f"<a:p>{run_text(text, size=size, color=color, bold=bold)}<a:endParaRPr lang=\"en-US\"/></a:p>"


def bullet(text: str, size: int = 19, level: int = 0) -> str:
    mar = 285750 + level * 285750
    return (
        f'<a:p><a:pPr marL="{mar}" indent="-171450"><a:buChar char="•"/></a:pPr>'
        f'{run_text(text, size=size)}<a:endParaRPr lang="en-US"/></a:p>'
    )


def tx_box(
    shape_id: int,
    x: float,
    y: float,
    w: float,
    h: float,
    paragraphs: list[str],
    fill: str | None = None,
    line: str | None = None,
) -> str:
    fill_xml = f'<a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>' if fill else "<a:noFill/>"
    line_xml = (
        f'<a:ln w="9525"><a:solidFill><a:srgbClr val="{line}"/></a:solidFill></a:ln>'
        if line
        else "<a:ln><a:noFill/></a:ln>"
    )
    body = "".join(paragraphs)
    return f"""
<p:sp>
  <p:nvSpPr><p:cNvPr id="{shape_id}" name="TextBox {shape_id}"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>
  <p:spPr><a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom>{fill_xml}{line_xml}</p:spPr>
  <p:txBody><a:bodyPr wrap="square" lIns="91440" tIns="45720" rIns="91440" bIns="45720"/><a:lstStyle/>{body}</p:txBody>
</p:sp>"""


def rect(shape_id: int, x: float, y: float, w: float, h: float, fill: str, line: str | None = None) -> str:
    line_xml = (
        f'<a:ln w="9525"><a:solidFill><a:srgbClr val="{line}"/></a:solidFill></a:ln>'
        if line
        else "<a:ln><a:noFill/></a:ln>"
    )
    return f"""
<p:sp>
  <p:nvSpPr><p:cNvPr id="{shape_id}" name="Band {shape_id}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>
  <p:spPr><a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>{line_xml}</p:spPr>
</p:sp>"""


def pic(shape_id: int, rel_id: str, name: str, x: float, y: float, w: float, h: float) -> str:
    return f"""
<p:pic>
  <p:nvPicPr><p:cNvPr id="{shape_id}" name="{esc(name)}"/><p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr><p:nvPr/></p:nvPicPr>
  <p:blipFill><a:blip r:embed="{rel_id}"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>
  <p:spPr><a:xfrm><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{emu(w)}" cy="{emu(h)}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr>
</p:pic>"""


def table_like(shape_id_start: int, x: float, y: float, col_w: list[float], row_h: float, rows: list[list[str]]) -> str:
    xml: list[str] = []
    sid = shape_id_start
    yy = y
    for ridx, row in enumerate(rows):
        xx = x
        for cidx, cell in enumerate(row):
            fill = "E7EEF7" if ridx == 0 else ("F8FAFC" if ridx % 2 else "FFFFFF")
            size = 15 if ridx else 14
            bold = ridx == 0
            xml.append(tx_box(sid, xx, yy, col_w[cidx], row_h, [paragraph(cell, size=size, bold=bold)], fill=fill, line="CBD5E1"))
            sid += 1
            xx += col_w[cidx]
        yy += row_h
    return "".join(xml)


def slide_xml(title: str, subtitle: str | None, body_shapes: list[str], section: str = "GeoNexus-RSD Progress") -> str:
    shapes = [
        rect(2, 0, 0, 13.333, 0.45, "8C1515"),
        tx_box(3, 0.45, 0.52, 8.8, 0.55, [paragraph(title, 26, "8C1515", True)]),
        tx_box(4, 9.6, 0.54, 3.0, 0.35, [paragraph(section, 10, "64748B")]),
    ]
    if subtitle:
        shapes.append(tx_box(5, 0.48, 1.05, 11.6, 0.35, [paragraph(subtitle, 14, "475569")]))
    shapes.extend(body_shapes)
    shapes.append(tx_box(980, 11.78, 7.10, 1.0, 0.25, [paragraph("", 8)]))
    return slide_package("".join(shapes))


def title_slide() -> str:
    shapes = [
        rect(2, 0, 0, 13.333, 7.5, "F8FAFC"),
        rect(3, 0, 0, 13.333, 0.62, "8C1515"),
        tx_box(4, 0.72, 1.45, 10.6, 0.9, [paragraph("GeoNexus-RSD", 40, "8C1515", True)]),
        tx_box(5, 0.75, 2.38, 10.9, 0.65, [paragraph("Hierarchy- and Context-Aware Vision-Language Prompting for Oriented Remote Sensing Detection", 20, "1F2937", True)]),
        tx_box(6, 0.78, 3.35, 5.2, 0.75, [paragraph("Progress report | DOTA2 first, DIOR-R validation", 18, "334155")], fill="FFFFFF", line="CBD5E1"),
        tx_box(7, 0.78, 4.65, 5.2, 0.95, [paragraph("Dinghao Li", 18, "1F2937", True), paragraph("Xi'an Jiaotong University", 15, "475569"), paragraph("June 13, 2026", 15, "475569")]),
        tx_box(8, 7.0, 1.55, 5.1, 3.8, [
            paragraph("Current Status", 20, "8C1515", True),
            bullet("DOTA2 S1 improves over RoI Transformer S0.", 17),
            bullet("S2 shows repeatable early-checkpoint gain, but final checkpoints are unstable.", 17),
            bullet("DIOR-R sanitized route is active with S1 replicas and S2 metrics pending.", 17),
        ], fill="FFFFFF", line="CBD5E1"),
    ]
    return slide_package("".join(shapes))


def slide_package(shape_tree: str) -> str:
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="{A_NS}" xmlns:r="{R_NS}" xmlns:p="{P_NS}">
  <p:cSld><p:spTree>
    <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
    <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{EMU_W}" cy="{EMU_H}"/><a:chOff x="0" y="0"/><a:chExt cx="{EMU_W}" cy="{EMU_H}"/></a:xfrm></p:grpSpPr>
    {shape_tree}
  </p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>'''


def agenda_slide() -> str:
    items = [("01", "Intro", "Problem and current GeoNexus-RSD route"),
             ("02", "Related Work", "Oriented detectors and RS vision-language prompting"),
             ("03", "Method", "Framework, prompt bank, regularization, adapters"),
             ("04", "Experiments", "DOTA2 and DIOR-R progress with current risks")]
    shapes = []
    y = 1.55
    sid = 10
    for num, head, desc in items:
        shapes.append(tx_box(sid, 0.9, y, 1.0, 0.7, [paragraph(num, 22, "8C1515", True)], fill="F1F5F9", line="CBD5E1"))
        shapes.append(tx_box(sid + 1, 2.1, y, 8.8, 0.7, [paragraph(head, 22, "1F2937", True), paragraph(desc, 14, "475569")]))
        y += 1.15
        sid += 2
    return slide_xml("Contents", "Advisor/group progress report structure", shapes)


SLIDES: list[tuple[str, str, list[str], list[tuple[str, str]]]] = []


def build_slide_defs() -> list[tuple[str, str, list[str], list[tuple[str, str]]]]:
    return [
        ("Intro: Oriented Detection Challenge", "Remote sensing detection remains sensitive to scale, rotation, density, and class granularity.", [
            tx_box(10, 0.7, 1.45, 5.1, 4.5, [
                paragraph("Why this setting is hard", 20, "8C1515", True),
                bullet("Objects appear at arbitrary orientations and large scale ranges.", 17),
                bullet("Fine-grained categories share similar textures and layouts.", 17),
                bullet("Dense scenes make proposal assignment and pseudo-label quality brittle.", 17),
                bullet("DOTA2 is now the formal benchmark route; DOTA v1.5 remains archive/debug evidence.", 17),
            ], fill="FFFFFF", line="CBD5E1"),
            pic(11, "rId2", "dotav2_contact_sheet_16x9.png", 6.25, 1.35, 5.9, 3.32),
            tx_box(12, 6.45, 4.95, 5.2, 0.65, [paragraph("Visual goal: improve prompt-aware oriented detection without over-claiming open-vocabulary behavior.", 15, "334155")], fill="F8FAFC", line="CBD5E1"),
        ], [("rId2", "media/geonexus_dotav2_contact_sheet_16x9.png")]),
        ("Intro: GeoNexus-RSD Route", "The current route is staged: stabilize detector evidence first, then reopen pseudo-label/context extensions.", [
            pic(10, "rId2", "route_progression_16x9.png", 0.75, 1.30, 5.9, 3.32),
            tx_box(11, 6.95, 1.38, 4.9, 3.8, [
                paragraph("Current gate", 20, "8C1515", True),
                bullet("DOTA2 S0 and S1 are completed and archived.", 17),
                bullet("S2 loss-0 has repeatable early-checkpoint improvement but unstable final checkpoints.", 17),
                bullet("DIOR-R sanitized S0 and S1 are available; S2 hierarchy replicas have clean startup and pending metrics.", 17),
                bullet("S3/S4, pseudo-label purification, FAIR1M, and routing stay paused.", 17),
            ], fill="FFFFFF", line="CBD5E1"),
        ], [("rId2", "media/geonexus_route_progression_16x9.png")]),
        ("Related Work: Detector and VLM Context", "We position GeoNexus against oriented detection baselines and recent RS prompt/OVD work.", [
            tx_box(10, 0.65, 1.35, 5.55, 4.8, [
                paragraph("Detector backbone", 20, "8C1515", True),
                bullet("RoI Transformer is the main controlled oriented detector for DOTA2 and DIOR-R.", 17),
                bullet("ORCNN and Rotated RetinaNet remain useful DIOR-R comparison baselines, but RoI Transformer currently leads.", 17),
                bullet("Evaluation uses DOTA-style mAP/AP50; best and final checkpoints are reported separately.", 17),
            ], fill="FFFFFF", line="CBD5E1"),
            tx_box(11, 6.55, 1.35, 5.55, 4.8, [
                paragraph("Vision-language references", 20, "8C1515", True),
                bullet("OpenRSD motivates open-prompt alignment for remote sensing detection.", 17),
                bullet("RemoteCLIP is used as the project VLM embedding source.", 17),
                bullet("RS-MPOD, DisDop, SOAR, and VK-Det inform future prompt, distillation, and pseudo-label ideas.", 17),
            ], fill="FFFFFF", line="CBD5E1"),
        ], []),
        ("Related Work: Project Gap", "Adjacent methods motivate the direction, but current claims must match implemented and measured modules.", [
            table_like(10, 0.55, 1.35, [2.05, 3.55, 4.7, 1.85], 0.62, [
                ["Source", "Main idea", "Relation to GeoNexus-RSD", "Use now"],
                ["OpenRSD", "Open prompts and alignment", "Reference framing for RS prompt detection", "Anchor"],
                ["RS-MPOD", "Multiple prompt cues", "Prompt ambiguity in aerial categories", "Defer"],
                ["SOAR / VK-Det", "Pseudo-label and prototype cues", "Relevant to planned S4 purification", "Paused"],
                ["DisDop", "Domain-prior distillation", "Possible future teacher-prior branch", "Future"],
            ]),
            tx_box(40, 0.7, 4.95, 11.2, 0.78, [paragraph("Conservative gap statement: current evidence is hierarchy/prompt-aware detector progress, not a completed open-vocabulary or pseudo-label purification system.", 16, "334155")], fill="F8FAFC", line="CBD5E1"),
        ], []),
        ("Method: Overall Framework", "Existing framework asset, interpreted as the current staged pipeline rather than a final end-to-end claim.", [
            pic(10, "rId2", "method_framework_16x9.png", 0.68, 1.25, 7.1, 4.0),
            tx_box(11, 8.05, 1.35, 3.95, 3.7, [
                paragraph("Implemented / active", 19, "8C1515", True),
                bullet("S0: RoI Transformer baseline", 16),
                bullet("S1: RemoteCLIP prompt scoring", 16),
                bullet("S2: hierarchy regularization variants", 16),
                paragraph("Planned / paused", 17, "8C1515", True),
                bullet("Context adapter and pseudo-label purification are not yet paper-facing claims.", 16),
            ], fill="FFFFFF", line="CBD5E1"),
        ], [("rId2", "media/geonexus_method_framework_16x9.png")]),
        ("Method Key Point: Prompt Bank and Scoring", "S1 uses a hierarchy-oriented prompt bank with RemoteCLIP-based prompt scoring.", [
            tx_box(10, 0.65, 1.35, 5.45, 4.65, [
                paragraph("Prompt bank", 20, "8C1515", True),
                bullet("Class prompts are organized around fine-grained RS object semantics.", 17),
                bullet("RemoteCLIP ViT-B/32 provides normalized prompt embeddings.", 17),
                bullet("The DOTA2 S2 prompt artifact validates 18 classes, finite embeddings, and an 18 x 18 relation matrix.", 17),
            ], fill="FFFFFF", line="CBD5E1"),
            tx_box(11, 6.55, 1.35, 5.45, 4.65, [
                paragraph("Detector integration", 20, "8C1515", True),
                bullet("RoI Transformer remains the detector backbone for controlled comparisons.", 17),
                bullet("S1 tests whether prompt scoring improves the DOTA2 detector route.", 17),
                bullet("Main DOTA2 S1 improves from 0.6088/0.6090 to 0.6177/0.6180.", 17),
            ], fill="FFFFFF", line="CBD5E1"),
        ], []),
        ("Method Key Point: Regularization and Planned Modules", "S2 is treated as an active stabilization question; later modules remain gated.", [
            tx_box(10, 0.65, 1.32, 3.65, 4.8, [
                paragraph("S2 hierarchy", 19, "8C1515", True),
                bullet("Regularizes class relations through hierarchy-aware targets.", 16),
                bullet("Loss-0 ablation gives repeatable early-checkpoint gains.", 16),
                bullet("Final-checkpoint instability blocks stronger claims.", 16),
            ], fill="FFFFFF", line="CBD5E1"),
            tx_box(11, 4.75, 1.32, 3.65, 4.8, [
                paragraph("Context adapter", 19, "8C1515", True),
                bullet("Planned scene/context prompt adapter.", 16),
                bullet("Useful for dense ports, airports, and urban scenes.", 16),
                bullet("Paused until S2 and DIOR-R stability are resolved.", 16),
            ], fill="FFFFFF", line="CBD5E1"),
            tx_box(12, 8.85, 1.32, 3.65, 4.8, [
                paragraph("Pseudo-label plan", 19, "8C1515", True),
                bullet("Purify pseudo labels with VLM and hierarchy cues.", 16),
                bullet("Inspired by SOAR/VK-Det/CastDet style evidence.", 16),
                bullet("Not part of current paper-facing results yet.", 16),
            ], fill="FFFFFF", line="CBD5E1"),
        ], []),
        ("Experiments: DOTA2 Progress", "DOTA2 is the primary benchmark route; S2 evidence is useful but not stable enough as a final claim.", [
            pic(10, "rId2", "dota2_baseline_lollipop_16x9.png", 0.65, 1.25, 5.6, 3.15),
            table_like(11, 6.55, 1.25, [2.45, 1.65, 1.65, 1.15], 0.56, [
                ["Stage", "mAP", "AP50", "Read"],
                ["S0 RoI Trans.", "0.6088", "0.6090", "base"],
                ["S1 GeoNexus", "0.6177", "0.6180", "up"],
                ["S2 best mean", "0.620606", "-", "early"],
                ["S2 final mean", "0.616655", "-", "unstable"],
            ]),
            tx_box(40, 0.8, 4.75, 10.9, 0.75, [paragraph("Interpretation: S1 is the clean current gain. S2 best-checkpoint behavior is repeatable, but final checkpoints fall slightly below S1 on average.", 15, "334155")], fill="F8FAFC", line="CBD5E1"),
        ], [("rId2", "media/geonexus_dota2_baseline_lollipop_16x9.png")]),
        ("Experiments: DIOR-R Sanitized Route", "DIOR-R moved from non-finite detector failures to sanitized finite training and active GeoNexus replicas.", [
            table_like(10, 0.55, 1.22, [3.15, 1.65, 1.55, 4.55], 0.58, [
                ["Run", "mAP", "AP50", "Status"],
                ["S0 RoI Transformer", "0.6531", "0.6530", "best archived on sanitized labels"],
                ["S1 replica A", "0.6751", "0.675", "completed"],
                ["S1 replica B", "0.6690", "0.669", "completed"],
                ["S2 hierarchy replicas", "pending", "pending", "launched with clean startup"],
            ]),
            tx_box(40, 0.7, 4.35, 11.1, 1.05, [
                paragraph("Readout", 18, "8C1515", True),
                bullet("Sanitization removed invalid-size records and bounded train-step diagnostics stayed finite.", 16),
                bullet("S1 improves over the sanitized RoI Transformer S0, but S2 hierarchy metrics are still active/pending.", 16),
            ], fill="FFFFFF", line="CBD5E1"),
        ], []),
    ]


def summary_slide() -> str:
    shapes = [
        rect(2, 0, 0, 13.333, 0.62, "8C1515"),
        tx_box(3, 0.7, 1.0, 10.8, 0.65, [paragraph("Summary and Next Steps", 30, "8C1515", True)]),
        tx_box(10, 0.75, 2.05, 3.65, 3.65, [
            paragraph("Completed", 20, "8C1515", True),
            bullet("DOTA2 S0/S1 archived.", 16),
            bullet("DOTA2 S2 loss-0 stability analyzed.", 16),
            bullet("DIOR-R sanitized S0/S1 evidence available.", 16),
        ], fill="FFFFFF", line="CBD5E1"),
        tx_box(11, 4.85, 2.05, 3.65, 3.65, [
            paragraph("Risks", 20, "8C1515", True),
            bullet("S2 final checkpoints remain unstable.", 16),
            bullet("DIOR-R S2 metrics are pending.", 16),
            bullet("Pseudo-label and context modules are not yet measured.", 16),
        ], fill="FFFFFF", line="CBD5E1"),
        tx_box(12, 8.95, 2.05, 3.65, 3.65, [
            paragraph("Next", 20, "8C1515", True),
            bullet("Archive DIOR-R S2 best/final metrics separately.", 16),
            bullet("Decide whether S2 is best-checkpoint evidence or stays diagnostic.", 16),
            bullet("Only then reopen context adapter and pseudo-label purification.", 16),
        ], fill="FFFFFF", line="CBD5E1"),
    ]
    return slide_package("".join(shapes))


def rels_xml(image_rels: list[tuple[str, str]]) -> str:
    rels = [
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout3.xml"/>'
    ]
    for rid, target in image_rels:
        rels.append(f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../{target}"/>')
    return f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="{REL_NS}">{"".join(rels)}</Relationships>'


def update_presentation(build: Path, num_slides: int) -> None:
    pres_path = build / "ppt" / "presentation.xml"
    tree = ET.parse(pres_path)
    root = tree.getroot()
    sld_id_lst = root.find(f"{{{P_NS}}}sldIdLst")
    assert sld_id_lst is not None
    sld_id_lst.clear()
    for i in range(1, num_slides + 1):
        elem = ET.SubElement(sld_id_lst, f"{{{P_NS}}}sldId", {"id": str(864 + i), f"{{{R_NS}}}id": f"rId{i + 1}"})
    tree.write(pres_path, encoding="UTF-8", xml_declaration=True)

    rel_path = build / "ppt" / "_rels" / "presentation.xml.rels"
    rel_root = ET.Element("Relationships", xmlns=REL_NS)
    ET.SubElement(rel_root, "Relationship", Id="rId1", Type=f"{R_NS}/slideMaster", Target="slideMasters/slideMaster1.xml")
    for i in range(1, num_slides + 1):
        ET.SubElement(rel_root, "Relationship", Id=f"rId{i + 1}", Type=f"{R_NS}/slide", Target=f"slides/slide{i}.xml")
    ET.SubElement(rel_root, "Relationship", Id=f"rId{num_slides + 2}", Type=f"{R_NS}/notesMaster", Target="notesMasters/notesMaster1.xml")
    ET.SubElement(rel_root, "Relationship", Id=f"rId{num_slides + 3}", Type=f"{R_NS}/presProps", Target="presProps.xml")
    ET.SubElement(rel_root, "Relationship", Id=f"rId{num_slides + 4}", Type=f"{R_NS}/viewProps", Target="viewProps.xml")
    ET.SubElement(rel_root, "Relationship", Id=f"rId{num_slides + 5}", Type=f"{R_NS}/theme", Target="theme/theme1.xml")
    ET.SubElement(rel_root, "Relationship", Id=f"rId{num_slides + 6}", Type=f"{R_NS}/tableStyles", Target="tableStyles.xml")
    ET.ElementTree(rel_root).write(rel_path, encoding="UTF-8", xml_declaration=True)


def update_content_types(build: Path, num_slides: int) -> None:
    ct_path = build / "[Content_Types].xml"
    tree = ET.parse(ct_path)
    root = tree.getroot()
    for elem in list(root):
        if elem.tag == f"{{{CT_NS}}}Override" and elem.attrib.get("PartName", "").startswith("/ppt/slides/"):
            root.remove(elem)
    for i in range(1, num_slides + 1):
        ET.SubElement(
            root,
            f"{{{CT_NS}}}Override",
            PartName=f"/ppt/slides/slide{i}.xml",
            ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml",
        )
    tree.write(ct_path, encoding="UTF-8", xml_declaration=True)


def copy_assets(build: Path) -> None:
    media = build / "ppt" / "media"
    mapping = {
        "dotav2_contact_sheet_16x9.png": "geonexus_dotav2_contact_sheet_16x9.png",
        "route_progression_16x9.png": "geonexus_route_progression_16x9.png",
        "method_framework_16x9.png": "geonexus_method_framework_16x9.png",
        "dota2_baseline_lollipop_16x9.png": "geonexus_dota2_baseline_lollipop_16x9.png",
    }
    for src, dst in mapping.items():
        shutil.copy2(ASSETS / src, media / dst)


def rebuild_package() -> None:
    build = BUILD_ROOT / f"_geonexus_xjtu_pptx_build_{os.getpid()}"
    if build.exists():
        shutil.rmtree(build)
    build.mkdir(parents=True)
    tmp_zip = build / "template.zip"
    shutil.copy2(TEMPLATE, tmp_zip)
    with zipfile.ZipFile(tmp_zip) as zf:
        zf.extractall(build)
    try:
        tmp_zip.unlink()
    except PermissionError:
        pass

    slide_dir = build / "ppt" / "slides"
    rel_dir = slide_dir / "_rels"

    copy_assets(build)
    slide_defs = build_slide_defs()
    slides = [(title_slide(), []), (agenda_slide(), [])]
    slides.extend((slide_xml(title, subtitle, shapes), rels) for title, subtitle, shapes, rels in slide_defs)
    slides.append((summary_slide(), []))
    assert len(slides) == 12

    for idx, (xml, img_rels) in enumerate(slides, 1):
        (slide_dir / f"slide{idx}.xml").write_text(xml, encoding="utf-8")
        (rel_dir / f"slide{idx}.xml.rels").write_text(rels_xml(img_rels), encoding="utf-8")

    update_presentation(build, len(slides))
    update_content_types(build, len(slides))

    if OUT.exists():
        OUT.unlink()
    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in build.rglob("*"):
            if path.is_file():
                if path == tmp_zip:
                    continue
                zf.write(path, posixpath.join(*path.relative_to(build).parts))


if __name__ == "__main__":
    rebuild_package()
    print(OUT)
