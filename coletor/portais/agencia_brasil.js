const puppeteer = require('puppeteer')
const {conectaBD, desconectaBD, insereNoticias} = require("./BD_handler")

async function coletaDadosAgenBr(pagina, link) {
  await pagina.goto(link, { waitUntil: "domcontentloaded" })
  let noticia = await pagina.evaluate(() => {
    const dados = {
      portal: {nome:"Agência Brasil"},
      _id: window.location.href
    }
  
    // Manchete
    let manchete = document.querySelector("h1.titulo-materia")
    if(manchete) dados.manchete = manchete.textContent.trim()
    else return null

    // Lide
    let lide = document.querySelector("div.linha-fina-noticia")
    if(lide) {
      if(lide.textContent.trim() != '') dados.lide = lide.textContent.trim()
    }
    // Data de Publicação
    let dataPublicacao = document.querySelector("meta[property='article:published_time']")
    if(dataPublicacao){
      dados.data = dataPublicacao.getAttribute("content")
    } else {
      return null
    }

    // Autores
    let autoresTag = document.querySelector(".autor-noticia")
    if(autoresTag) {
        let autores = autoresTag.textContent
        autores = autores.split('–')[0].trim()
        autores = autores.split('-')[0].trim()
        autores = autores.replace(/ e /g,",")
        autores = autores.split(',')
        dados.autor = autores.filter(x => x.length > 0).map(x => {
          const autor = {nome: x.trim()}
          return autor
        })
    }

    // Tags
    let tags = Array.from(document.querySelectorAll("a.tag")).map(x => x.textContent.toUpperCase())
    if(tags && tags.length > 0) dados.assunto = tags.map(x => {
      const tag = {}
      tag.TAG = x
      return tag
    })

    return dados
  })
  if(noticia != null) {
    noticia.data = new Date(noticia.data)
    if((noticia.data instanceof Date) == false) noticia = null
  }
  return noticia
}



async function scrapAgenciaBrasil(URL, tipo) {
  const browser = await puppeteer.launch()
  const scrapingPage = await browser.newPage()
  const paginaPortal = await browser.newPage()
  const {db, client} = await conectaBD() 
  let dict = []

  try {
    for (let pagina = 1; pagina <= 999; pagina++) {
      let AgenciaURL = `${URL}${pagina}`
      await paginaPortal.bringToFront()
      await paginaPortal.goto(AgenciaURL, { waitUntil: "domcontentloaded" })

      const links = await paginaPortal.evaluate(() => {
        return Array.from(document.querySelectorAll(".capa-noticia")).map(x => x.getAttribute("href"))
      })


      let raiz = "https://agenciabrasil.ebc.com.br"
      for(let i = 0; i < links.length; i++ ){
        links[i] = raiz + links[i]
      }

      await scrapingPage.bringToFront()
      for (let i = 0; i < links.length; i++) {
        let temp = await coletaDadosAgenBr(scrapingPage, links[i])
        if(temp == null) continue;
        temp.secao = tipo
        dict.push(temp)
      }
      let inseridos = await insereNoticias(dict, db)
      if(inseridos < 0.5) {
        return null
      }
      dict = []
    }
  } catch (err) {
    return
    console.error("Erro:", err)
  } finally {
    await desconectaBD(client)
    await scrapingPage.close()
    await browser.close()
  }
}

async function scrapingAgenciaBrasil(){
  console.log('Coletando Agência Brasil...')
  await Promise.all([    
    scrapAgenciaBrasil("https://agenciabrasil.ebc.com.br/economia?page=", "Economia"),
    scrapAgenciaBrasil("https://agenciabrasil.ebc.com.br/politica?page=", "Política")
  ])
}

module.exports = {scrapingAgenciaBrasil}