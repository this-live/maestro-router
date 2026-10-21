# Architecture diagrams

Two diagrams describe Maestro 3.1.1.

- [architecture.svg](architecture.svg) shows the runtime. The command line and the HTTP server both hand a task to the router, which classifies it, walks the policy ladder and maps the result onto the models you can run (the model map). It picks a local endpoint, a cloud seat or a specialist. An optional JEV classifier and local feedback learning sit alongside.
- [dataflow.svg](dataflow.svg) shows the same path as five stages, from the task text through classify, policy and inventory to the chosen seat.
