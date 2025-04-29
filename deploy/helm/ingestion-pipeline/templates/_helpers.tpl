{{/*
Expand the name of the chart.
*/}}

{{- define "ingestion-pipeline.name" -}}
{{- printf "%s-v%s" .Values.name .Values.version | replace "." "-"  | replace " " "-" }}
{{- end }}

{{- define "ingestion-pipeline.pipelineName" -}}
{{- default .Release.Namespace | replace "_" "-" | trimSuffix "-" }}
{{- end }}