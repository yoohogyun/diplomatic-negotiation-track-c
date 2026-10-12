$ErrorActionPreference = 'Stop'
$taskRoot = 'C:\Users\user\Desktop\26-2\EMNLP\MIDTERM\diplomatic-negotiation-fresh'
$pptxPath = Join-Path $taskRoot 'release\diplomatic-negotiation-track-c\paper\figures\teaser_editable_v2.pptx'
$pdfPath = Join-Path $taskRoot 'tmp\teaser_pptx\teaser_native.pdf'
$pngPath = Join-Path $taskRoot 'tmp\teaser_pptx\teaser_native.png'
$pptApp = New-Object -ComObject PowerPoint.Application
$deck = $null
try {
  $deck = $pptApp.Presentations.Open($pptxPath, -1, 0, 0)
  $deck.SaveAs($pdfPath, 32)
  $deck.Slides.Item(1).Export($pngPath, 'PNG', 2520, 840)
  Write-Output "Exported editable slide to PDF and PNG. Slides: $($deck.Slides.Count)"
} finally {
  if ($null -ne $deck) { $deck.Close() }
  $pptApp.Quit()
  [void][Runtime.InteropServices.Marshal]::ReleaseComObject($pptApp)
}

