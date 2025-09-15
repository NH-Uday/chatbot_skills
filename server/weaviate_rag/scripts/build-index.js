// Load environment variables
const dotenv = require("dotenv");
dotenv.config();

const fs = require("fs");
const path = require("path");
const pdfParse = require("pdf-parse");
const { OpenAI } = require("openai");

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const DOC_DIR = process.env.DOC_DIR || path.join(__dirname, "../docs");
const OUT = path.join(__dirname, "../data/docs_index.json");
const EMBED_MODEL = process.env.EMBED_MODEL || "text-embedding-3-small";
const CHUNK_SIZE = 1200; // characters
const CHUNK_OVERLAP = 150; // characters

// Split text into overlapping chunks
// Split text into overlapping chunks (robust)
function chunkText(text) {
  const chunks = [];
  const n = text.length;

  // If the page is small, just return it as one chunk
  if (n === 0) return [];
  if (n <= CHUNK_SIZE) return [text.trim()];

  // Ensure we always make progress, even if overlap >= chunk size
  const step = Math.max(1, CHUNK_SIZE - CHUNK_OVERLAP);

  let i = 0;
  while (i < n) {
    const end = Math.min(n, i + CHUNK_SIZE);
    chunks.push(text.slice(i, end));
    if (end === n) break;     // reached the end
    i += step;                // advance by (chunk - overlap)
  }

  // Cleanup whitespace
  return chunks
    .map(t => t.replace(/\s+\n/g, "\n").trim())
    .filter(Boolean);
}


// Embed an array of text chunks
async function embedAll(payloads) {
  const inputs = payloads.map(p => p.text);
  const resp = await openai.embeddings.create({
    model: EMBED_MODEL,
    input: inputs
  });
  return resp.data.map(d => d.embedding);
}

async function run() {
  console.log(`📄 Reading PDFs from: ${DOC_DIR}`);
  const files = fs.readdirSync(DOC_DIR).filter(f => /\.pdf$/i.test(f));

  if (!files.length) {
    console.error("❌ No PDF files found in doc folder.");
    process.exit(1);
  }

  const records = [];

  // Process each PDF
  for (const file of files) {
    console.log(`➡️ Processing ${file}...`);
    const fullPath = path.join(DOC_DIR, file);
    const data = await pdfParse(fs.readFileSync(fullPath));
    const pages = data.text.split("\f"); // Page breaks

    for (let p = 0; p < pages.length; p++) {
      const pageText = pages[p].trim();
      if (!pageText) continue;
      const chunks = chunkText(pageText);
      for (let j = 0; j < chunks.length; j++) {
        records.push({
          id: `${file}::${p + 1}::${j}`,
          file,
          page: p + 1,
          text: chunks[j]
        });
      }
    }
  }

  console.log(`🧩 Total chunks: ${records.length}`);

  // Embed in batches
  const BATCH_SIZE = 100;
  let all = [];
  for (let i = 0; i < records.length; i += BATCH_SIZE) {
    const batch = records.slice(i, i + BATCH_SIZE);
    console.log(`🔢 Embedding batch ${i / BATCH_SIZE + 1}...`);
    const embs = await embedAll(batch);
    for (let k = 0; k < batch.length; k++) {
      all.push({
        ...batch[k],
        embedding: embs[k]
      });
    }
  }

  fs.mkdirSync(path.dirname(OUT), { recursive: true });
  fs.writeFileSync(OUT, JSON.stringify(all, null, 2));
  console.log(`✅ Index built and saved to ${OUT}`);
}

// Run
run().catch(err => {
  console.error("❌ Error building index:", err);
  process.exit(1);
});
