/**
 * Fetch all documents from Stanford SearchWorks for a given collection.
 * Run this in your browser console while on searchworks.stanford.edu
 */

async function fetchAllDocs(
  collectionId,
  formatType = "Archive/Manuscript",
  perPage = 100
) {
  const baseUrl = "https://searchworks.stanford.edu/catalog.json"
  const allDocs = []
  let page = 1

  while (true) {
    // Build URL with parameters
    const params = new URLSearchParams({
      "f[collection][]": collectionId,
      "f[format_hsim][]": formatType,
      page: page,
      per_page: perPage
    })

    const url = `${baseUrl}?${params.toString()}`

    console.log(`Fetching page ${page}...`)

    try {
      // Make the request
      const response = await fetch(url)

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      // Parse JSON response
      const data = await response.json()

      // Extract docs from this page
      const docs = data?.response?.docs || []
      allDocs.push(...docs)

      console.log(
        `  Retrieved ${docs.length} documents (total so far: ${allDocs.length})`
      )

      // Check if this is the last page
      const pagesInfo = data?.response?.pages || {}
      const isLastPage = pagesInfo["last_page?"] ?? true

      if (isLastPage) {
        console.log(`Reached last page (page ${page})`)
        break
      }

      // Move to next page
      page++

      // Add a small delay to be polite to the server
      await new Promise(resolve => setTimeout(resolve, 500))
    } catch (error) {
      console.error(`Error fetching page ${page}:`, error)
      break
    }
  }

  return allDocs
}

/**
 * Download the combined documents as a JSON file
 */
function downloadJSON(data, filename = "combined_docs.json") {
  const jsonStr = JSON.stringify(data, null, 2)
  const blob = new Blob([jsonStr], { type: "application/json" })
  const url = URL.createObjectURL(blob)

  const a = document.createElement("a")
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)

  console.log(`Downloaded ${filename}`)
}

/**
 * Main execution function
 */
async function scrapeStanfordArchive() {
  // Collection ID as a parameter
  const COLLECTION_ID = "in00000122003"

  console.log(`Starting scrape for collection: ${COLLECTION_ID}`)

  // Fetch all documents
  const allDocuments = await fetchAllDocs(COLLECTION_ID)

  // Print summary
  console.log(`\nTotal documents retrieved: ${allDocuments.length}`)

  // Print first document as example
  if (allDocuments.length > 0) {
    console.log("\nExample - First document keys:")
    console.log(Object.keys(allDocuments[0]))
    console.log("\nFirst document sample:")
    console.log(allDocuments[0])
  }

  // Store in a global variable for access
  window.stanfordDocs = allDocuments
  console.log("\nDocuments stored in: window.stanfordDocs")

  // Prompt to download
  const shouldDownload = confirm(
    `Found ${allDocuments.length} documents. Download as JSON?`
  )
  if (shouldDownload) {
    downloadJSON(allDocuments)
  }

  return allDocuments
}

// Auto-run the scraper
console.log("Stanford Archive Scraper loaded!")
console.log("Run: scrapeStanfordArchive()")

// Uncomment the next line to auto-run when pasted:
// scrapeStanfordArchive();
