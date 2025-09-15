// index.js
import express from "express";
import cors from "cors";
import helmet from "helmet";
import morgan from "morgan";
import fs from "fs";
import dotenv from "dotenv";
dotenv.config();

import { OpenAI } from "openai";

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const app = express();
const PORT = process.env.PORT || 3001;
const MODEL = process.env.MODEL || "gpt-4o-mini";
const EMBED_MODEL = process.env.EMBED_MODEL || "text-embedding-3-small";
const INDEX_PATH = process.env.INDEX_PATH || "./data/docs_index.json";
const TOP_K = Number(process.env.TOP_K || 6);
const MIN_SIM = Number(process.env.MIN_SIM || 0.80); // refusal threshold

app.use(helmet());
app.use(express.json({ limit: "1mb" }));
app.use(cors());
app.use(morgan("tiny"));

app.get("/health", (_, res) => res.json({ ok: true }));

// --- Load index once ---
let INDEX = [];
try {
  INDEX = JSON.parse(fs.readFileSync(INDEX_PATH, "utf8"));
  console.log(`Loaded index with ${INDEX.length} chunks`);
} catch (e) {
  console.warn("No local index found. Build it with scripts/build-index.js");
}

// --- Cosine similarity ---
function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-8);
}

// --- Language hint from last user message (DE/EN) ---
function languageHint(messages) {
  const last = [...messages].reverse().find(m => m.role === "user");
  const txt = (last?.content || "").toLowerCase();
  const isGerman = /[äöüß]|( der | die | das | und | ist | nicht | warum | wie )/.test(` ${txt} `);
  return isGerman ? "de" : "en";
}

const SYSTEM_RAG = `
You are an inspiring AI tutor for Computational Fluid Dynamics and Fluid Mechanics.
Answer ONLY using the provided "COURSE CONTEXT". If the question is outside this context, reply with a refusal.

Formatting & behavior:
- Respond in the SAME language as the user's question (Deutsch ODER English).
- Use **three bold-titled sections with emojis**:
  **🧠 Clarity**, **⚖️ Contrast (if applicable)**, **🚀 Motivational Close**.
- Use proper mathematical notation with LaTeX in Markdown:
  inline: \\( ... \\), display: \\[ ... \\].
- When explaining equations, break them down term-by-term.
- Cite snippets inline like [p. <page>, <file>].
- If the topic is outside the materials, say:
  (DE) "Ich bin nicht dafür ausgelegt, Fragen außerhalb der bereitgestellten Vorlesungsunterlagen zu beantworten."
  (EN) "I’m not designed to answer questions outside the provided lecture materials."
`;

app.post("/chat", async (req, res) => {
  try {
    const { messages } = req.body || {};
    if (!Array.isArray(messages) || messages.length === 0) {
      return res.status(400).json({ error: "Body must include non-empty 'messages' array." });
    }

    const userText = [...messages].reverse().find(m => m.role === "user")?.content || "";
    // 1) Embed query
    const qEmb = await openai.embeddings.create({ model: EMBED_MODEL, input: userText });
    const qVec = qEmb.data[0].embedding;

    // 2) Retrieve top chunks
    const scored = INDEX.map(r => ({ r, sim: cosine(qVec, r.embedding) }))
      .sort((a, b) => b.sim - a.sim)
      .slice(0, TOP_K);

    const bestSim = scored[0]?.sim ?? 0;

    // 3) Refusal if below threshold
    const lang = languageHint(messages);
    if (!scored.length || bestSim < MIN_SIM) {
      const refusal = lang === "de"
        ? "Ich bin nicht dafür ausgelegt, Fragen außerhalb der bereitgestellten Vorlesungsunterlagen zu beantworten."
        : "I’m not designed to answer questions outside the provided lecture materials.";
      return res.json({ reply: { role: "assistant", content: `**🧠 Clarity**\n${refusal}\n\n**⚖️ Contrast (if applicable)**\n—\n\n**🚀 Motivational Close**\n- Frage mich gerne zu den Vorlesungsinhalten (CFD, Strömungsmechanik, Diskretisierung, Konvektion-Diffusion, etc.).` } });
    }

    // 4) Build course context with citations
    const contextBlocks = scored.map(({ r, sim }) =>
      `[#${r.id}] (sim=${sim.toFixed(3)}) [p. ${r.page}, ${r.file}]\n${r.text}`
    ).join("\n\n---\n\n");

    const userMsg = `
COURSE CONTEXT:
${contextBlocks}

USER QUESTION:
${userText}

RULE: Use only the COURSE CONTEXT. If information is missing, say you cannot answer outside the materials.
`;

    // 5) Call Responses API (single shot, non-streaming)
    const resp = await openai.responses.create({
      model: MODEL,
      temperature: 0.4,
      input: [
        { role: "system", content: SYSTEM_RAG },
        { role: "user", content: userMsg },
      ],
    });

    // 6) Extract text
    let text = "";
    for (const item of resp.output ?? []) {
      if (item.type === "message") {
        for (const c of item.content ?? []) {
          if (c.type === "output_text") text += c.text;
        }
      }
    }
    text = text.trim();

    return res.json({ reply: { role: "assistant", content: text } });
  } catch (err) {
    console.error(err);
    const msg = err?.error?.message || err?.message || "Failed to generate response";
    return res.status(500).json({ error: msg });
  }
});

app.listen(PORT, () => console.log(`Server running at http://localhost:${PORT}`));
