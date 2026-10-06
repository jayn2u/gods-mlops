"use strict";

function csrfToken() {
  const prefix = "gods_mlops_operator_csrf=";
  const item = document.cookie.split(";").map((value) => value.trim()).find((value) => value.startsWith(prefix));
  return item ? decodeURIComponent(item.slice(prefix.length)) : "";
}

function bindCsrfForms() {
  document.querySelectorAll("form[data-csrf-form]").forEach((form) => {
    if (form.dataset.csrfBound === "true") return;
    form.dataset.csrfBound = "true";
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      if (form.dataset.submitting === "true") return;
      form.dataset.submitting = "true";
      const submitButtons = form.querySelectorAll('button[type="submit"], input[type="submit"]');
      submitButtons.forEach((button) => {
        button.disabled = true;
      });
      try {
        const response = await fetch(form.action, {
          method: form.method.toUpperCase(),
          body: new FormData(form),
          credentials: "same-origin",
          headers: { "X-CSRF-Token": csrfToken() },
        });
        if (response.redirected) {
          window.location.assign(response.url);
          return;
        }
        if (response.status === 202) {
          document.documentElement.innerHTML = await response.text();
          bindCsrfForms();
          return;
        }
        if (response.ok) {
          window.location.reload();
          return;
        }
        document.documentElement.innerHTML = await response.text();
        bindCsrfForms();
      } catch (error) {
        form.dataset.submitting = "false";
        submitButtons.forEach((button) => {
          button.disabled = false;
        });
        throw error;
      }
    });
  });
}

bindCsrfForms();
