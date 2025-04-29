{{/*
Expand the name of the chart.
*/}}
{{- define "configure-pipeline.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}
