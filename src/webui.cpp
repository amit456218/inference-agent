#include "webui.h"

const char* const kWebUI = R"HTML(<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ingot chat</title>
<style>
  :root { color-scheme: light dark; --bg:#fff; --fg:#111; --muted:#777; --user:#e8f0fe; --bot:#f3f3f3; --line:#ddd; --accent:#2563eb; }
  @media (prefers-color-scheme: dark) { :root { --bg:#151515; --fg:#eee; --muted:#999; --user:#1f2d4a; --bot:#232323; --line:#333; --accent:#5b8cff; } }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.5 -apple-system, system-ui, sans-serif; display:flex; flex-direction:column; height:100vh; }
  header { display:flex; gap:12px; align-items:center; padding:10px 16px; border-bottom:1px solid var(--line); flex-wrap:wrap; }
  header b { font-size:16px; }
  header span { color:var(--muted); font-size:13px; }
  header label { font-size:13px; color:var(--muted); display:flex; align-items:center; gap:4px; }
  header input[type=number] { width:56px; }
  header input[type=text] { width:220px; }
  header button { margin-left:auto; }
  #log { flex:1; overflow-y:auto; padding:16px; display:flex; flex-direction:column; gap:10px; }
  .msg { max-width:80%; padding:10px 14px; border-radius:14px; white-space:pre-wrap; word-wrap:break-word; }
  .user { background:var(--user); align-self:flex-end; }
  .assistant { background:var(--bot); align-self:flex-start; }
  .stats { font-size:12px; color:var(--muted); align-self:flex-start; margin-top:-6px; }
  form { display:flex; gap:8px; padding:12px 16px; border-top:1px solid var(--line); }
  textarea { flex:1; resize:none; font:inherit; padding:8px 10px; border:1px solid var(--line); border-radius:8px; background:var(--bg); color:var(--fg); }
  button { font:inherit; padding:8px 14px; border:0; border-radius:8px; background:var(--accent); color:#fff; cursor:pointer; }
  button.secondary { background:transparent; color:var(--muted); border:1px solid var(--line); }
  button:disabled { opacity:.5; cursor:default; }
</style></head><body>
<header>
  <b>Ingot chat</b><span id="model">connecting...</span>
  <label>system <input type="text" id="system" placeholder="(optional) You are a helpful assistant."></label>
  <label>temp <input type="number" id="temp" value="0.7" min="0" max="2" step="0.1"></label>
  <label>max tokens <input type="number" id="max" value="512" min="1" step="64"></label>
  <button class="secondary" id="clear" type="button">New chat</button>
</header>
<div id="log"></div>
<form id="f"><textarea id="in" rows="2" placeholder="Message (Enter to send, Shift+Enter for newline)"></textarea><button id="send">Send</button></form>
<script>
const log = document.getElementById('log'), input = document.getElementById('in'), send = document.getElementById('send');
let messages = [], busy = false;
fetch('/v1/models').then(r => r.json()).then(j => document.getElementById('model').textContent = j.data[0].id).catch(() => document.getElementById('model').textContent = 'server unreachable');
function add(role, text) { const d = document.createElement('div'); d.className = 'msg ' + role; d.textContent = text; log.appendChild(d); log.scrollTop = log.scrollHeight; return d; }
document.getElementById('clear').onclick = () => { messages = []; log.innerHTML = ''; input.focus(); };
input.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); document.getElementById('f').requestSubmit(); } });
document.getElementById('f').onsubmit = async e => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text || busy) return;
  busy = true; send.disabled = true; input.value = '';
  messages.push({ role: 'user', content: text });
  add('user', text);
  const bubble = add('assistant', '');
  const sys = document.getElementById('system').value.trim();
  const body = { messages: (sys ? [{ role: 'system', content: sys }] : []).concat(messages), stream: true,
                 temperature: parseFloat(document.getElementById('temp').value), max_tokens: parseInt(document.getElementById('max').value) };
  const t0 = performance.now(); let n = 0, reply = '';
  try {
    const res = await fetch('/v1/chat/completions', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    if (!res.ok) throw new Error((await res.json()).error.message);
    const reader = res.body.getReader(), dec = new TextDecoder(); let buf = '';
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf('\n\n')) >= 0) {
        const line = buf.slice(0, i).trim(); buf = buf.slice(i + 2);
        if (!line.startsWith('data: ') || line === 'data: [DONE]') continue;
        const delta = JSON.parse(line.slice(6)).choices[0].delta.content;
        if (delta) { reply += delta; n++; bubble.textContent = reply; log.scrollTop = log.scrollHeight; }
      }
    }
    messages.push({ role: 'assistant', content: reply });
    const s = (performance.now() - t0) / 1000;
    const st = document.createElement('div'); st.className = 'stats'; st.textContent = n + ' tokens, ' + (n / s).toFixed(1) + ' tok/s'; log.appendChild(st);
  } catch (err) { bubble.textContent = 'error: ' + err.message; messages.pop(); }
  busy = false; send.disabled = false; input.focus(); log.scrollTop = log.scrollHeight;
};
input.focus();
</script></body></html>
)HTML";
