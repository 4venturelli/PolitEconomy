const puppeteer = require('puppeteer')
const { conectaBD, desconectaBD, insereNoticias } = require("./BD_handler")


async function coletaDadosRevistaForum(pagina, link) {
  await pagina.goto(link, { waitUntil: "domcontentloaded" })
  const noticia = await pagina.evaluate(() => {
    const dados = {
      portal: {nome:"Revista Fórum"},
      _id: window.location.href,
    }

    // Manchete 
    const manchete = document.querySelector("h1.titulo")
    if (manchete) dados.manchete = manchete.textContent.trim()
      else return null

    // Lide
    const lide = document.querySelector("h2.bajada")
    if (lide) dados.lide = lide.textContent.trim()

    // Data de publicação
    const dataPublicacao = document.querySelector('time.fecha-time')
    if (dataPublicacao) {
      let data = dataPublicacao.getAttribute('datetime')
      data = data.split("T")
      let dia = data[0]
      let hora = data[1]
      dia = dia.split('-')
      dia = dia.map(x => {
        if(x.length < 2) {
          x = "0".concat(x)
        }
        return x
      })
      hora = hora.split(':')
      hora = hora.map(x => {
        if(x.length < 2) {
          x = "0".concat(x)
        }
        return x
      })
      dia = dia.join('-')
      hora = hora.join(':')
      dados.data = dia.concat("T".concat(hora.concat("-03:00")))

    } else {
      return null
    }

    // Autores
    const autoresTag = document.querySelector('span.post-author-name')
    if (autoresTag) {
      let autores = autoresTag.textContent.trim(); 
      let array = autores.split(/[,\/]/).map(a => a.trim()).filter(a => a.length > 0);
      if(array.length > 0) {
        dados.autor = array.map(x => {
          let autor = {nome: x}
          return autor
        })
      }
    }

    //Tags
    let tags = document.querySelector("meta[name='keywords']")
    if(tags) {
      let tagsTexto = tags.getAttribute('content')
      tagsTexto = tagsTexto.split(',')
      dados.assunto = tagsTexto.map(x => {
        let tag = {TAG: x.trim().toUpperCase()}
        return tag
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



async function scrapRevistaForum(URL, secao) {
  const browser = await puppeteer.launch({headless:true})
  const scrapingPage = await browser.newPage()
  const paginaPortal = await browser.newPage()
  const {db, client} = await conectaBD()
  await paginaPortal.goto(`${URL}`, { waitUntil: "domcontentloaded" })

  let dict = []

  try{
    for(let i = 1; i <= 800; i++){
      await paginaPortal.bringToFront()
      // links
      let links = await paginaPortal.evaluate(() => {
        return Array.from(document.querySelectorAll("h2.titulo a")).map(el => "https://revistaforum.com.br".concat(el.getAttribute("href")))
      })
  
      // remove os links antigos
      await paginaPortal.evaluate(() => {
        const artigosAntigos = document.querySelectorAll('.caja')
        artigosAntigos.forEach(artigo => artigo.remove())
      });
    
      // clica no botão
      try {
        let clickResult = await paginaPortal.locator('div.btn').click({count: 2 ,delay: 1000})
      } catch (e) {
        return null
          console.log("Não foi possível carregar novos conteúdos")
          console.log(e)
      }   
      await new Promise(resolve => setTimeout(resolve, 1000))
      
      for (let i = 0; i < links.length; i++) {
        let temp = await coletaDadosRevistaForum(scrapingPage, links[i])

        if(temp == null) continue;
        temp.secao = secao
        dict.push(temp)

      }

      let inseridos = await insereNoticias(dict, db)
      if(inseridos < 0.5) {
          return null
      }
      dict = []
    }
    if(dict.length > 0) insereNoticias(dict, db)

  } catch (err) {
    return
    console.error("Erro:", err)
  } finally {
    await desconectaBD(client)
    await scrapingPage.close()
    await browser.close()
  }
    
}

async function scrapingRevistaForum(){
  console.log('Coletando Revista Fórum...')
  await Promise.all([
    scrapRevistaForum("https://revistaforum.com.br/economia/", "Economia"),
    scrapRevistaForum("https://revistaforum.com.br/politica/", "Política")
  ])
}

module.exports = {scrapingRevistaForum}