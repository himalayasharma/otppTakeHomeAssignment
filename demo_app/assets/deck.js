(function () {
  function clickButton(id) {
    var button = document.getElementById(id);
    if (button && !button.disabled) {
      button.click();
    }
  }

  document.addEventListener("keydown", function (event) {
    var tagName = event.target && event.target.tagName;
    if (["INPUT", "TEXTAREA", "SELECT", "BUTTON"].indexOf(tagName) !== -1) {
      return;
    }

    if (event.key === "ArrowRight" || event.key === "PageDown") {
      event.preventDefault();
      clickButton("next-scene");
    }

    if (event.key === "ArrowLeft" || event.key === "PageUp") {
      event.preventDefault();
      clickButton("prev-scene");
    }
  });
})();
