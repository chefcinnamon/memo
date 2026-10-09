const test = require('node:test');
const assert = require('node:assert/strict');
const { spawn, spawnSync } = require('node:child_process');
const { createHash } = require('node:crypto');
const fs = require('node:fs/promises');
const net = require('node:net');
const os = require('node:os');
const path = require('node:path');

const repository = path.resolve(__dirname, '..');
const hugo = process.env.HUGO_BIN || 'hugo';
const hasHugo = spawnSync(hugo, ['version']).status === 0;

async function eventually(read, accept) {
  const deadline = Date.now() + 12000;
  let lastError;
  while (Date.now() < deadline) {
    try {
      const value = await read();
      if (accept(value)) return value;
    } catch (error) { lastError = error; }
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  throw new Error(`Hugo did not rebuild in time${lastError ? ': ' + lastError.message : ''}`);
}

test('LP assets survive live CSS and shortcode rebuilds', { skip: !hasHugo, timeout: 45000 }, async () => {
  const fixture = await fs.mkdtemp(path.join(os.tmpdir(), 'memo-lp-assets-'));
  let server;
  let log = '';
  async function write(relative, content) {
    const target = path.join(fixture, relative);
    await fs.mkdir(path.dirname(target), { recursive: true });
    await fs.writeFile(target, content);
  }
  try {
    for (const relative of ['layouts/_partials/head.html', 'layouts/_partials/head/css.html',
      'layouts/_partials/head/favicon.html', 'layouts/shortcodes/lp-playground.html',
      'assets/css/lp-playground.css', 'assets/js/lp-playground.js']) {
      await write(relative, await fs.readFile(path.join(repository, relative)));
    }
    await write('hugo.toml', 'baseURL = "http://localhost/"\n[markup.goldmark.renderer]\nunsafe = true\n');
    await write('layouts/_default/baseof.html', '<!doctype html><html><head>{{ partial "head.html" . }}</head><body>{{ block "main" . }}{{ end }}</body></html>');
    await write('layouts/_default/single.html', '{{ define "main" }}{{ .Content }}{{ end }}');
    await write('content/lp.md', '+++\ntitle = "LP"\n+++\n{{< lp-playground mode="signs" >}}\n{{< lp-playground mode="position" >}}\n');
    await write('content/plain.md', '+++\ntitle = "Plain"\n+++\nNo playground.\n');

    const reservation = net.createServer();
    await new Promise(resolve => reservation.listen(0, '127.0.0.1', resolve));
    const port = reservation.address().port;
    await new Promise(resolve => reservation.close(resolve));
    const origin = `http://127.0.0.1:${port}`;
    server = spawn(hugo, ['server', '--source', fixture, '--bind', '127.0.0.1',
      '--port', String(port), '--baseURL', origin + '/', '--appendPort=false', '--disableFastRender'],
      { stdio: ['ignore', 'pipe', 'pipe'] });
    server.stdout.on('data', chunk => { log += chunk; });
    server.stderr.on('data', chunk => { log += chunk; });
    const page = async () => {
      const response = await fetch(origin + '/lp/');
      assert.equal(response.status, 200);
      return response.text();
    };
    function assets(html) {
      return [...html.matchAll(/<(?:link|script)\b[^>]*(?:href|src)="([^"]*lp-playground[^"]*)"[^>]*integrity="([^"]+)"[^>]*>/g)];
    }
    async function verify(html) {
      assert.equal((html.match(/data-lp-playground=/g) || []).length, 2);
      const tags = assets(html);
      assert.equal(tags.filter(tag => tag[1].endsWith('.js')).length, 1);
      assert.equal(tags.filter(tag => tag[1].endsWith('.css')).length, 1);
      for (const [, url, integrity] of tags) {
        const response = await fetch(new URL(url, origin));
        assert.equal(response.status, 200);
        const bytes = Buffer.from(await response.arrayBuffer());
        const decodedIntegrity = integrity.replace(/&#(x[0-9a-f]+|\d+);/gi, (_, value) =>
          String.fromCodePoint(value[0].toLowerCase() === 'x' ? parseInt(value.slice(1), 16) : Number(value)));
        assert.equal(decodedIntegrity, 'sha256-' + createHash('sha256').update(bytes).digest('base64'));
      }
    }
    const initial = await eventually(page, html => assets(html).length === 2);
    await verify(initial);
    assert.equal(assets(await (await fetch(origin + '/plain/')).text()).length, 0);

    const cssURL = assets(initial).find(tag => tag[1].endsWith('.css'))[1];
    await fs.appendFile(path.join(fixture, 'assets/css/lp-playground.css'), '\n.lp-playground { --rebuild-probe: 1; }\n');
    const afterCSS = await eventually(page, html => assets(html).some(tag => tag[1].endsWith('.css') && tag[1] !== cssURL));
    await verify(afterCSS);

    await fs.appendFile(path.join(fixture, 'layouts/shortcodes/lp-playground.html'), '\n<span data-shortcode-rebuild-probe hidden></span>\n');
    const afterTemplate = await eventually(page, html => html.includes('shortcode-rebuild-probe'));
    await verify(afterTemplate);
  } catch (error) {
    error.message += '\n' + log;
    throw error;
  } finally {
    if (server && server.exitCode === null) {
      const exited = new Promise(resolve => server.once('exit', resolve));
      server.kill('SIGTERM');
      await exited;
    }
    await fs.rm(fixture, { recursive: true, force: true });
  }
});
