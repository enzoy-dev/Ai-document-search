"use strict";

const el = (id) => document.getElementById(id);
let settings = null;
let key = "";
let pdfUrl = null;
let busy = false;

function showError(error) {
  el("error").textContent =
    error.message || "Não foi possível concluir a operação.";
  el("error").hidden = false;
}

async function request(path, options = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 180000);
  try {
    const headers = { ...(options.headers || {}) };
    if (!settings?.demo_mode && key) headers["X-API-Key"] = key;
    const response = await fetch(path, {
      ...options,
      headers,
      signal: controller.signal,
      cache: "no-store",
    });
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      const messages = {
        401: "A chave de acesso não foi aceita. Confira e conecte novamente.",
        409: "Já existe um documento com esse nome na coleção.",
        413: "O PDF ultrapassa o limite de tamanho permitido.",
        429: "O limite de uso da demonstração foi atingido. Tente novamente mais tarde.",
        503: "O serviço está temporariamente indisponível. A busca de trechos pode continuar disponível se apenas a geração falhou.",
      };
      throw new Error(
        messages[response.status] ||
          (typeof data.detail === "string"
            ? data.detail
            : "Não foi possível concluir a operação."),
      );
    }
    return response;
  } catch (error) {
    if (error.name === "AbortError")
      throw new Error(
        "A operação demorou mais que o esperado. Atualize a coleção antes de tentar novamente.",
      );
    if (error instanceof TypeError)
      throw new Error(
        "Não foi possível conectar ao serviço. Confira sua conexão e tente novamente.",
      );
    throw error;
  } finally {
    clearTimeout(timer);
  }
}

function node(tag, className, text) {
  const item = document.createElement(tag);
  if (className) item.className = className;
  if (text !== undefined) item.textContent = text;
  return item;
}

function clearPdf() {
  el("pdf-viewer").removeAttribute("src");
  el("pdf-download").removeAttribute("href");
  if (pdfUrl) URL.revokeObjectURL(pdfUrl);
  pdfUrl = null;
  el("pdf-panel").hidden = true;
}

async function openPdf(filename, page = 1) {
  try {
    el("error").hidden = true;
    const prefix = settings.demo_mode ? "/demo" : "";
    const response = await request(
      `${prefix}/documents/${encodeURIComponent(filename)}/pdf`,
    );
    const blob = await response.blob();
    clearPdf();
    pdfUrl = URL.createObjectURL(blob);
    el("pdf-title").textContent = `${filename} · página ${page || 1}`;
    el("pdf-download").href = pdfUrl;
    el("pdf-download").download = filename;
    el("pdf-viewer").src = `${pdfUrl}#page=${Number(page) || 1}`;
    el("pdf-panel").hidden = false;
    el("pdf-panel").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    showError(error);
  }
}

async function loadDocuments() {
  const prefix = settings.demo_mode ? "/demo" : "";
  const { documents } = await (await request(`${prefix}/documents`)).json();
  el("document-list").replaceChildren();
  el("document-count").textContent = documents.length;
  el("document-help").textContent = documents.length
    ? "Selecione um PDF para conferir o original."
    : "Adicione um PDF com texto para começar.";
  for (const document of documents) {
    const row = node("div", "document-row");
    const button = node("button", "document-button");
    button.append(
      node("span", "document-icon", "▤"),
      node("span", "", document.filename),
    );
    button.setAttribute("aria-label", `Abrir ${document.filename}`);
    button.addEventListener("click", () => openPdf(document.filename));
    row.append(button);
    if (!settings.demo_mode) {
      const remove = node("button", "delete-button", "×");
      remove.setAttribute("aria-label", `Excluir ${document.filename}`);
      remove.addEventListener("click", async () => {
        if (
          !window.confirm(
            `Excluir ${document.filename} e seus trechos da coleção?`,
          )
        )
          return;
        try {
          await request(`/documents/${encodeURIComponent(document.filename)}`, {
            method: "DELETE",
          });
          clearPdf();
          el("results").hidden = true;
          await loadDocuments();
        } catch (error) {
          showError(error);
        }
      });
      row.append(remove);
    }
    el("document-list").append(row);
  }
  el("upload-form").hidden = settings.demo_mode;
}

function renderResults(data, mode) {
  const retrieve = mode === "retrieve";
  const sources = data.sources || [];
  const titles = {
    answered: "Resposta com referências",
    not_found: "Informação não encontrada",
    needs_clarification: "Preciso de mais um detalhe",
  };
  el("result-title").textContent = retrieve
    ? sources.length
      ? "Trechos encontrados"
      : "Nenhum trecho próximo encontrado"
    : titles[data.status];
  el("result-kind").textContent = retrieve
    ? "Busca sem geração"
    : "Resposta gerada";
  el("answer").textContent = retrieve
    ? sources.length
      ? "Confira se os trechos abaixo contêm a informação que você procura. A proximidade da busca não garante que eles respondam à pergunta."
      : "Tente uma pergunta mais específica ou consulte outro documento."
    : data.answer;
  el("sources").replaceChildren();
  for (const source of sources) {
    const card = node("article", "source-card");
    const header = node("div", "source-header");
    const label = node("div", "source-label");
    label.append(
      node("span", "source-id", `[${source.citation_id}]`),
      document.createTextNode(
        `${source.filename} · página ${source.page ?? "não registrada"}`,
      ),
    );
    const button = node("button", "text-link", "Conferir no PDF");
    button.addEventListener("click", () =>
      openPdf(source.filename, source.page),
    );
    header.append(label, button);
    card.append(header, node("p", "source-text", source.text));
    el("sources").append(card);
  }
  el("results").hidden = false;
}

el("connection-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  key = el("api-key").value.trim();
  el("api-key").value = "";
  clearPdf();
  el("results").hidden = true;
  el("error").hidden = true;
  try {
    await loadDocuments();
  } catch (error) {
    key = "";
    el("upload-form").hidden = true;
    el("document-list").replaceChildren();
    el("document-count").textContent = "0";
    showError(error);
  }
});

el("upload-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = el("pdf-file").files[0];
  if (!file) return;
  if (file.size > settings.max_upload_size_mb * 1024 * 1024) {
    showError(new Error("O arquivo ultrapassa o limite de tamanho permitido."));
    return;
  }
  const button = event.currentTarget.querySelector("button");
  button.disabled = true;
  el("upload-progress").hidden = false;
  el("error").hidden = true;
  try {
    const data = new FormData();
    data.append("file", file);
    await request("/documents/upload", { method: "POST", body: data });
    el("upload-form").reset();
    await loadDocuments();
  } catch (error) {
    showError(error);
  } finally {
    button.disabled = false;
    el("upload-progress").hidden = true;
  }
});

el("question-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (busy) return;
  if (!settings?.demo_mode && !key) {
    showError(
      new Error(
        "Conecte sua coleção com a chave de acesso antes de pesquisar.",
      ),
    );
    return;
  }
  const question = el("question").value.trim();
  if (!question) {
    showError(new Error("Escreva uma pergunta para pesquisar."));
    return;
  }
  const mode = document.querySelector(
    'input[name="search-mode"]:checked',
  ).value;
  busy = true;
  el("ask-button").disabled = true;
  el("search-progress").hidden = false;
  el("results").hidden = true;
  el("results").setAttribute("aria-busy", "true");
  el("error").hidden = true;
  clearPdf();
  try {
    const prefix = settings.demo_mode ? "/demo" : "";
    const query = new URLSearchParams({ question });
    const data = await (await request(`${prefix}/${mode}?${query}`)).json();
    renderResults(data, mode);
  } catch (error) {
    showError(error);
  } finally {
    busy = false;
    el("ask-button").disabled = false;
    el("search-progress").hidden = true;
    el("results").setAttribute("aria-busy", "false");
  }
});

el("close-pdf").addEventListener("click", clearPdf);
window.addEventListener("pagehide", clearPdf);

async function start() {
  try {
    settings = await (await request("/ui-config")).json();
    el("generation-mode").disabled = !settings.generation_enabled;
    el("generation-help").textContent = settings.generation_enabled
      ? "Buscar trechos não utiliza geração de respostas. Selecione a opção ao lado para responder com referências."
      : "Nesta configuração, você pode buscar trechos e conferir os PDFs. A geração de respostas está desativada.";
    el("upload-limit").textContent =
      `PDFs com texto, até ${settings.max_upload_size_mb} MB. Documentos digitalizados precisam de OCR antes do envio.`;
    if (settings.demo_mode) {
      el("connection-form").hidden = true;
      el("mode-badge").textContent = "Demonstração";
      el("notice").textContent =
        `Explore documentos fictícios, sem enviar arquivos pessoais. A coleção é compartilhada e aceita até ${settings.requests_per_minute} consultas por minuto no total.`;
      el("notice").hidden = false;
      el("examples").hidden = false;
      for (const question of settings.example_questions) {
        const button = node("button", "example-button", question);
        button.addEventListener("click", () => {
          el("question").value = question;
          el("question").focus();
        });
        el("example-buttons").append(button);
      }
      await loadDocuments();
    }
  } catch (error) {
    showError(error);
    el("ask-button").disabled = true;
  }
}
start();
