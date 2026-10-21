"""Rebuild the original fictional holdout PDFs and labels using only pypdf."""

import json
from pathlib import Path
import textwrap

from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject

ROOT = Path(__file__).resolve().parent / "holdout"
DOCUMENTS = {
    "reembolsos.pdf": [
        "Política fictícia de reembolsos — Cooperativa Horizonte\n"
        "Vigência: 1 de março de 2025. Esta política cobre viagens de trabalho previamente aprovadas.\n"
        "Alimentação: limite de R$ 180 por pessoa por dia. Bebidas alcoólicas não são reembolsadas.\n"
        "Hospedagem: limite de R$ 250 por diária, mediante nota fiscal.\n"
        "O deslocamento em ônibus pode ser reembolsado com comprovante da passagem.",
        "Procedimento de solicitação — Cooperativa Horizonte\n"
        "Envie a nota fiscal e a identificação da viagem pelo portal interno em até 10 dias corridos após o retorno.\n"
        "A liderança da equipe aprova a solicitação. O financeiro paga em até 7 dias úteis após a aprovação.\n"
        "Despesas acima dos limites exigem autorização prévia por escrito.\n"
        "Não há reembolso automático para despesas sem comprovante.",
    ],
    "operacao_atlas.pdf": [
        "Runbook fictício — Serviço Atlas\n"
        "Janela de manutenção: quintas-feiras, das 02h00 às 02h20 UTC.\n"
        "O alerta de latência dispara após três verificações consecutivas acima de 2 segundos.\n"
        "A equipe Plataforma é responsável pelo primeiro atendimento.\n"
        "Durante a manutenção, o serviço mostra uma página de indisponibilidade programada.",
        "Backup e recuperação — Serviço Atlas\n"
        "O backup completo é executado diariamente às 03h00 UTC. A retenção é de 14 dias.\n"
        "O objetivo de perda máxima de dados é de 24 horas. O objetivo de recuperação é de 35 minutos.\n"
        "Antes de restaurar, a equipe valida o checksum e registra o incidente.\n"
        "Este documento não contém senhas nem credenciais de produção.",
    ],
    "observatorio.pdf": [
        "Agenda fictícia — Observatório Serra Clara\n"
        "As visitas públicas acontecem às sextas-feiras, das 19h às 22h, no horário de Brasília.\n"
        "A entrada é gratuita. São 24 vagas por sessão. A idade mínima é de 12 anos.\n"
        "A reserva deve ser feita com pelo menos 2 dias de antecedência.\n"
        "Em noites de chuva, a visita é substituída por uma apresentação no auditório.",
        "Oficina de astronomia — Observatório Serra Clara\n"
        "A oficina ocorre aos sábados às 15h, no horário de Brasília, e dura 90 minutos.\n"
        "O conteúdo é apresentado em português. A atividade inclui uma demonstração de mapas celestes.\n"
        "Participantes recebem material impresso ao final da oficina.\n"
        "A oficina tem inscrição separada da visita noturna.",
    ],
}


def write_pdf(path, pages):
    writer = PdfWriter()
    font = writer._add_object(
        DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
                NameObject("/Encoding"): NameObject("/WinAnsiEncoding"),
            }
        )
    )
    for text in pages:
        page = writer.add_blank_page(width=612, height=792)
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
        )
        lines = [
            line
            for paragraph in text.splitlines()
            for line in textwrap.wrap(paragraph, width=82)
        ]
        escaped = [
            line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            for line in lines
        ]
        stream = DecodedStreamObject()
        stream.set_data(
            (
                "BT /F1 11 Tf 45 745 Td 17 TL\n"
                + "\n".join(f"({line}) Tj T*" for line in escaped)
                + "\nET"
            ).encode("cp1252")
        )
        page[NameObject("/Contents")] = writer._add_object(stream)
    writer.add_metadata(
        {
            "/Title": path.stem,
            "/Author": "AI Document Search — fictional evaluation fixture",
        }
    )
    writer.write(path)


def main():
    directory = ROOT / "documents"
    directory.mkdir(parents=True, exist_ok=True)
    for name, pages in DOCUMENTS.items():
        write_pdf(directory / name, pages)
    labeled = [
        ("Qual é o limite diário de alimentação?", "reembolsos.pdf", 1, ["180"]),
        ("Qual é o limite por diária de hospedagem?", "reembolsos.pdf", 1, ["250"]),
        ("Bebidas alcoólicas são reembolsadas?", "reembolsos.pdf", 1, ["não"]),
        (
            "Em quantos dias devo solicitar o reembolso após retornar?",
            "reembolsos.pdf",
            2,
            ["10"],
        ),
        (
            "Quanto tempo o financeiro tem para pagar após a aprovação?",
            "reembolsos.pdf",
            2,
            ["7", "úteis"],
        ),
        ("What is the daily meal reimbursement limit?", "reembolsos.pdf", 1, ["180"]),
        (
            "Quando ocorre a manutenção do serviço Atlas?",
            "operacao_atlas.pdf",
            1,
            ["02", "UTC"],
        ),
        (
            "Quantas verificações de latência acionam o alerta?",
            "operacao_atlas.pdf",
            1,
            ["três"],
        ),
        ("Por quantos dias os backups são retidos?", "operacao_atlas.pdf", 2, ["14"]),
        (
            "Qual é o objetivo de recuperação do Atlas em minutos?",
            "operacao_atlas.pdf",
            2,
            ["35"],
        ),
        (
            "What is the maximum data-loss objective for Atlas?",
            "operacao_atlas.pdf",
            2,
            ["24"],
        ),
        (
            "Em quais dias acontecem as visitas públicas ao observatório?",
            "observatorio.pdf",
            1,
            ["sextas"],
        ),
        ("Qual é a capacidade de uma sessão de visita?", "observatorio.pdf", 1, ["24"]),
        (
            "Qual é a idade mínima para visitar o observatório?",
            "observatorio.pdf",
            1,
            ["12"],
        ),
        ("Quanto tempo dura a oficina de astronomia?", "observatorio.pdf", 2, ["90"]),
        (
            "What language is the astronomy workshop presented in?",
            "observatorio.pdf",
            2,
            ["Portuguese"],
        ),
    ]
    cases = [
        {
            "id": f"holdout-{i:02}",
            "kind": "answerable",
            "question": q,
            "expected": [{"filename": name, "page": page}],
            "answer_keywords": keywords,
            "expected_status": "answered",
        }
        for i, (q, name, page, keywords) in enumerate(labeled, 1)
    ]
    for kind, questions in {
        "unanswerable": [
            "Qual é a senha de produção do Atlas?",
            "Qual é o salário da direção do observatório?",
            "What is the reimbursement portal URL?",
            "Qual é o telefone do observatório?",
        ],
        "ambiguous": [
            "Qual é o limite?",
            "Qual é o prazo?",
            "Quando acontece a atividade?",
            "What is the schedule?",
        ],
    }.items():
        for q in questions:
            cases.append(
                {
                    "id": f"holdout-{len(cases) + 1:02}",
                    "kind": kind,
                    "question": q,
                    "expected": [],
                    "expected_status": "needs_clarification"
                    if kind == "ambiguous"
                    else "not_found",
                }
            )
    dataset = {
        "version": 2,
        "description": "Original fictional holdout: policy, technical runbook and public schedule. Created after the original model comparison; no tuning on this set. Small diagnostic, not a general accuracy estimate.",
        "documents": [f"documents/{name}" for name in DOCUMENTS],
        "cases": cases,
    }
    (ROOT / "retrieval.json").write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
