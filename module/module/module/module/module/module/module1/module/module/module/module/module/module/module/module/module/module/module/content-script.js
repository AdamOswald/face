// ==UserScript==
// @name         Craiyon Download Button
// @namespace    http://tampermonkey.net/
// @version      0.1
// @description  try to take over the world!
// @author       You
// @match        *.craiyon.com/*
// @icon         data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==
// @grant        none
// ==/UserScript==

(function () {
  "use strict";
  document.onreadystatechange = () => {
    console.log("READY!!!!!!");
    setTimeout(() => {
      console.log("START!!!!!");
      let scrsh = document.querySelector('button[aria-label="Screenshot"]');
      let button = scrsh.cloneNode();
      scrsh.parentElement.appendChild(button);

      let svg = document.createElement("svg");
      button.append(svg);
      svg.outerHTML =
        '<svg aria-hidden="true" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg" class="mr-1 h-6 w-6 text-white"><path d="M9.557 13.562a.619.619 0 0 0 .443.183.631.631 0 0 0 .443-.183l3.123-3.117a.624.624 0 1 0-.882-.884l-2.059 2.053v-8.49a.624.624 0 1 0-1.25 0v8.49L7.316 9.561a.624.624 0 1 0-.882.884l3.123 3.117zm7.943-1.06v3.749c0 .688-.56 1.249-1.25 1.249H3.75c-.69 0-1.25-.561-1.25-1.25v-3.748a.624.624 0 1 1 1.25 0v3.749h12.5v-3.749a.624.624 0 1 1 1.25 0z" clip-rule="evenodd" fill-rule="evenodd"></path></svg>';
      //button.textContent = "Download all 9 pictures";//textContent makes svg disappear

      let span = document.createElement("span");
      button.append(span);
      span.outerHTML =
        '<span class="hidden sm:flex">Download all 9 pictures</span>';

      button.onclick = () => {
        let is = document.querySelectorAll("img.h-full");
        let a = document.createElement("a");
        document.body.append(a);
        let title = document.querySelector("#prompt").innerText;
        is.forEach((v, i) => {
          a.href = v.src;
          a.setAttribute("download", title + ` ${i + 1}`);
          a.click();
        });
      };
    }, 0);
  };
})();

// ==UserScript==
// @name         Craiyon Download Button
// @namespace    http://tampermonkey.net/
// @version      0.1
// @description  try to take over the world!
// @author       You
// @match        *.craiyon.com/*
// @icon         data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==
// @grant        none
// ==/UserScript==

(function () {
  "use strict";
  document.onreadystatechange = () => {
    console.log("READY!!!!!!");
    setTimeout(() => {
      console.log("START!!!!!");
      let scrsh = document.querySelector('button[aria-label="Screenshot"]');
      let button = scrsh.cloneNode();
      scrsh.parentElement.appendChild(button);

      let svg = document.createElement("svg");
      button.append(svg);
      svg.outerHTML =
        '<svg aria-hidden="true" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg" class="mr-1 h-6 w-6 text-white"><path d="M9.557 13.562a.619.619 0 0 0 .443.183.631.631 0 0 0 .443-.183l3.123-3.117a.624.624 0 1 0-.882-.884l-2.059 2.053v-8.49a.624.624 0 1 0-1.25 0v8.49L7.316 9.561a.624.624 0 1 0-.882.884l3.123 3.117zm7.943-1.06v3.749c0 .688-.56 1.249-1.25 1.249H3.75c-.69 0-1.25-.561-1.25-1.25v-3.748a.624.624 0 1 1 1.25 0v3.749h12.5v-3.749a.624.624 0 1 1 1.25 0z" clip-rule="evenodd" fill-rule="evenodd"></path></svg>';
      //button.textContent = "Download all 9 pictures";//textContent makes svg disappear

      let span = document.createElement("span");
      button.append(span);
      span.outerHTML =
        '<span class="hidden sm:flex">Download all 9 pictures</span>';

      button.onclick = () => {
        let is = document.querySelectorAll("img.h-full");
        let a = document.createElement("a");
        document.body.append(a);
        let title = document.querySelector("#prompt").innerText;
        is.forEach((v, i) => {
          a.href = v.src;
          a.setAttribute("download", title + ` ${i + 1}`);
          a.click();
        });
      };
    }, 0);
  };
})();

