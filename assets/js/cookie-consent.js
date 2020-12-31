function loadGA1onConsent() {
    window.ga=function(){ga.q.push(arguments)};ga.q=[];ga.l=+new Date;
    ga('create','{{ site.analytics.google.tracking_id }}','auto');
    ga('set', 'anonymizeIp', false);
    ga('send','pageview')
}

function loadGA2onConsent() {
    window.dataLayer = window.dataLayer || [];
    function gtag(){dataLayer.push(arguments);}
    gtag('js', new Date());

    gtag('config', '{{ site.analytics.google.tracking_id }}', { 'anonymize_ip': {{ site.analytics.google.anonymize_ip | default: false }}});
 }
 
function loadGA3onConsent() {
    (function(c,l,a,r,i,t,y){
        c[a]=c[a]||function(){(c[a].q=c[a].q||[]).push(arguments)};
        t=l.createElement(r);t.async=1;t.src="https://www.clarity.ms/tag/"+i;
        y=l.getElementsByTagName(r)[0];y.parentNode.insertBefore(t,y);
    })(window, document, "clarity", "script", "4q5jrakqqa");
}
window.cookieconsent.initialise({
  "palette": {
    "popup": {
      "background": "#000"
    },
    "button": {
      "background": "#f1d600"
    }
  },
  "type": "opt-in",
  "content": {
    "href": "https://blackbird71sr.github.io/terms"
  },
  onInitialise: function (status) {
    var type = this.options.type;
    var didConsent = this.hasConsented();
    if (type == 'opt-in' && didConsent) {
      // enable cookies
      loadGA2onConsent();
      loadGA3onConsent();
    }
    if (type == 'opt-out' && !didConsent) {
      // disable cookies
    }
  },
  onStatusChange: function(status, chosenBefore) {
    var type = this.options.type;
    var didConsent = this.hasConsented();
    if (type == 'opt-in' && didConsent) {
      // enable cookies
      loadGA2onConsent();
      loadGA3onConsent();
    }
    if (type == 'opt-out' && !didConsent) {
      // disable cookies
    }
  },
  onRevokeChoice: function() {
    var type = this.options.type;
    if (type == 'opt-in') {
      // disable cookies
    }
    if (type == 'opt-out') {
      // enable cookies
      loadGA2onConsent();
      loadGA3onConsent();
    }
  }
});
