const puppeteer = require('puppeteer')
const {conectaBD, desconectaBD, insereNoticias} = require("./BD_handler")

async function coletaDadosG1(pagina, link) {
  await pagina.goto(link, { waitUntil: "domcontentloaded" })
  const noticia = await pagina.evaluate(() => {
    const dados = {
      portal: {nome:"g1"},
      _id : window.location.href
    }

    // Manchete
    let manchete = document.querySelector("h1.content-head__title")
    if(manchete) dados.manchete = manchete.textContent
    else return null
    
    // Lide
    let lide = document.querySelector("h2.content-head__subtitle")
    if(lide) dados.lide = lide.textContent
    
    // Data
    let dataPublicacao = document.querySelector('time[itemprop="dateModified"]')
    if(dataPublicacao) {
      dados.data = dataPublicacao.getAttribute("datetime")
    }
    else return null
      
    // Autores
    let autoresTag = document.querySelector("p.top__signature__text__author-name")
    if(autoresTag == null) {
      autoresTag = document.querySelector("p.content-publication-data__from")
    }
    if(autoresTag) {
      let autores = autoresTag.textContent
      autores = autores.replace("Por", '')
      if(autores.indexOf(" —") >= 0) autores = autores.slice(0, autores.indexOf(" —")) // remove o traço e a localização que vem depois dele.
      autores = autores.split(',')
      if(autores[autores.length - 1].indexOf(" e " >= 0)) {
        let dupla = autores.pop()
        dupla = dupla.split(" e ")
        for(let i = 0; i < dupla.length; i++) autores.push(dupla[i])
      }
      dados.autor = autores.map(x => {
        let autor = {nome: x.trim()}
        return autor
      })
    }
    return dados
  })
if(noticia != null) {
    noticia.data = new Date(noticia.data)
    if((noticia.data instanceof Date) == false) noticia = null
  }
  return noticia
}

async function scrapG1(URL, secao) {
  const browser = await puppeteer.launch({headless : true})
  const paginaScraping = await browser.newPage()
  const paginaPortal = await browser.newPage()
  const {db, client} = await conectaBD()

  let dict = []
  
  try {
    for (let pagina = 1; pagina <= 999; pagina++) {
      let g1URL = `${URL}${pagina}.ghtml`
      await paginaPortal.bringToFront()
      await paginaPortal.goto(g1URL, { waitUntil: "domcontentloaded" })

      let links = await paginaPortal.evaluate(() => {
        return Array.from(document.querySelectorAll(".feed-post-link")).map(x => x.getAttribute("href"))
      })
      

      await paginaScraping.bringToFront()
      for (let i = 0; i < links.length; i++) {
        let temp = undefined
        try {
          temp = await coletaDadosG1(paginaScraping, links[i])
        } catch(err) {
          let tentativas = 1
          console.log(err)
          while(typeof temp === "undefined" && tentativas <= 5){
            temp = await coletaDadosG1(paginaScraping, links[i])
            tentativas++
          }
          if(typeof temp === "undefined") {
            if(dict.length > 0) insereNoticias(dict, db)
            return null
          }
        }
        if(temp == null) continue

        temp.secao = secao
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
    await paginaScraping.close()
    await browser.close()
  }
}



async function scrapingG1(){
  console.log('Coletando g1...')
  await Promise.all([
    scrapG1("https://g1.globo.com/economia/index/feed/pagina-", "Economia"),
    scrapG1(`https://g1.globo.com/politica/index/feed/pagina-`, "Política")
  ])
}

module.exports = {scrapingG1}