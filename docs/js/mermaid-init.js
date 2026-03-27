// Material theme with fence_code_format emits <pre class="mermaid"><code>...</code></pre>.
// Extract textContent (which decodes HTML entities like --&gt; to -->) and re-render
// as a plain <div class="mermaid"> that Mermaid can process.
document.addEventListener('DOMContentLoaded', function () {
  document.querySelectorAll('pre.mermaid').forEach(function (pre) {
    var div = document.createElement('div');
    div.className = 'mermaid';
    div.textContent = pre.querySelector('code')
      ? pre.querySelector('code').textContent
      : pre.textContent;
    pre.replaceWith(div);
  });
  mermaid.initialize({ startOnLoad: false, theme: 'default' });
  mermaid.run({ querySelector: '.mermaid' });
});
