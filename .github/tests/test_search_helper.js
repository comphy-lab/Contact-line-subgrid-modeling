'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const helperPath = path.resolve(__dirname, '../assets/js/search-helper.js');
let searchResults = [];
const context = {
  URL,
  document: { querySelector: () => null },
  console: { error: () => {}, log: () => {}, warn: () => {} },
  window: {
    location: {
      href: 'https://comphy-lab.org/Contact-line-subgrid-modeling/index.html',
      origin: 'https://comphy-lab.org'
    },
    searchData: []
  }
};

context.Fuse = class {
  search() {
    return searchResults.slice();
  }
};

vm.createContext(context);
vm.runInContext(fs.readFileSync(helperPath, 'utf8'), context, {
  filename: helperPath
});

const resolveNavigationUrl = context.window.searchHelper.resolveNavigationUrl;
assert.equal(
  resolveNavigationUrl('../gle-ode/gle-solve.c.html'),
  'https://comphy-lab.org/gle-ode/gle-solve.c.html'
);
assert.equal(
  resolveNavigationUrl('https://comphy-lab.org/other-project/page.html'),
  'https://comphy-lab.org/other-project/page.html'
);
assert.equal(resolveNavigationUrl('javascript:alert(1)'), null);
assert.equal(resolveNavigationUrl('data:text/html,<script>alert(1)</script>'), null);
assert.equal(resolveNavigationUrl('https://example.org/redirect'), null);
assert.equal(resolveNavigationUrl('//example.org/redirect'), null);
assert.equal(resolveNavigationUrl(null), null);

(async () => {
  const originalHref = context.window.location.href;
  searchResults = [{
    refIndex: 0,
    score: 0,
    item: {
      title: 'Unsafe result',
      excerpt: 'Must not navigate',
      url: 'javascript:alert(1)'
    }
  }];
  let commands = await context.window.searchHelper.searchDatabaseForCommandPalette('unsafe');
  assert.equal(commands.length, 1);
  commands[0].handler();
  assert.equal(context.window.location.href, originalHref);

  searchResults = [{
    refIndex: 1,
    score: 0,
    item: {
      title: 'Safe result',
      excerpt: 'Navigate within the site',
      url: '/Contact-line-subgrid-modeling/src-local/gle-model.h.html'
    }
  }];
  commands = await context.window.searchHelper.searchDatabaseForCommandPalette('safe');
  commands[0].handler();
  assert.equal(
    context.window.location.href,
    'https://comphy-lab.org/Contact-line-subgrid-modeling/src-local/gle-model.h.html'
  );

  console.log('search-helper URL validation: verified');
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
