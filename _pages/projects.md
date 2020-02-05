---
layout: archive
permalink: /projects/
title: "Projects"
author_profile: true
header:
  image: "/images/singapore.JPG"
---

{% for post in site.posts %}
  {% include archive-single.html %}
{% endfor %}