// Copyright (c) Microsoft. All rights reserved. Licensed under the MIT license. See LICENSE file in the project root for full license information.
var opCommon = require('./op.common.js');

exports.transform = function (model) {
  model.layout = model.layout || "Reference";
  model._op_layout = model.layout;
  model.pagetype = "Reference";
  // Clean up unused predefined properties
  var resetKeys = [
    "attributes",
    "uid",
    "id",
    "parent",
    "children",
    "href",
    "name",
    "fullName",
    "type",
    "source",
    "documentation",
    "assemblies",
    "namespace",
    "summary",
    "remarks",
    "example",
    "syntax",
    "overridden",
    "exceptions",
    "seealso",
    "see",
    "inheritance",
    "level",
    "implements",
    "inheritedMembers",
    "conceptual",
    "platform",
    "newFileRepository",
    "thread_safety",
    "defined_in",
    "supported_platforms",
    "requirements",
    "isEii",
    "isExtensionMethod",
    "nameWithType",
    "extensionMethods",
    "_baseDirectory",
    "_displayLangs",
    "gitContribute",
    "_gitContribute",
    "langs",
    "derivedClasses"
  ];

  model = opCommon.resetKeysAndSystemAttributes(model, resetKeys, true);
  return {
    content: JSON.stringify(model)
  };
}
