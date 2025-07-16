const puppeteer = require('puppeteer')
const {conectaBD, desconectaBD, insereNoticias} = require("./BD_handler")

async function coletaDadosBBC(pagina, link) {
  await pagina.goto(link, { waitUntil: "domcontentloaded" })
  let noticia = await pagina.evaluate(() => {
    const dados = {
      portal: {nome:"BBC"},
      _id: window.location.href
    }
  
    // Manchete
    let manchete = document.querySelector('h1#content')
    if(manchete) dados.manchete = manchete.textContent.trim()
    else return null

    // Lide
    let lide = document.querySelector('[data-testid="caption-paragraph"]')
    if(lide && lide.textContent.trim().length > 0) dados.lide = lide.textContent.trim()

    // Data de Publicação
    let dataPublicacao = document.querySelector("meta[name='article:published_time']")
    if(dataPublicacao){
      dados.data = dataPublicacao.getAttribute("content")
    } else {
      return null
    }

    // Autores
    let autoresTag = document.querySelector("span.bbc-1ypcc2")
    if(autoresTag) {
        let autores = autoresTag.textContent
        autores = autores.replace(/ e /g,",")
        autores = autores.split(',')
        dados.autor = autores.filter(x => x.length > 0).map(x => {
          const autor = {nome: x.trim()}
          return autor
        })
    }

    // Tags
    let tags = Array.from(document.querySelectorAll("meta[name='article:tag']")).map(x => x.getAttribute("content").toUpperCase())
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



async function scrapBBC(URL, secao) {
  const browser = await puppeteer.launch({headless: true})
  const scrapingPage = await browser.newPage()
  const paginaPortal = await browser.newPage()
  const {db, client} = await conectaBD() 
  let dict = []

  try {
    for (let pagina = 1; pagina <= 40; pagina++) {
      let BBCURL = `${URL}${pagina}`
      await paginaPortal.bringToFront()
      await paginaPortal.goto(BBCURL, { waitUntil: "domcontentloaded" })

      const links = await paginaPortal.evaluate(() => {
        return Array.from(document.querySelectorAll("h2.bbc-1slyjq2 a")).map(x => x.getAttribute("href"))
      })

      await scrapingPage.bringToFront()
      
      for (let i = 0; i < links.length; i++) {
        let temp = undefined;
        try {
          temp = await coletaDadosBBC(scrapingPage, links[i])
        } catch(e) {
          console.log(e)
          let tentativas = 1
          while(typeof temp === "undefined" && tentativas <= 5) {
            temp = await coletaDadosBBC(scrapingPage, links[i])
            tentativas++
          }
          if(typeof temp === "undefined") {
            if(dict.length > 0) insereNoticias(dict, db)
            return null
          }
        }
        if(temp == null){
            continue
        }  
        temp.secao = secao
        dict.push(temp)
      }

      let inseridos = await insereNoticias(dict, db);
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

async function scrapingBBC(){
  console.log('Coletando BBC...')
  await Promise.all([
    scrapBBC("https://www.bbc.com/portuguese/topics/cg7267qwzx1t?page=", "Política"),
    scrapBBC("https://www.bbc.com/portuguese/topics/cvjp2jr0k9rt?page=", "Economia")
  ])
}

module.exports = {scrapingBBC}