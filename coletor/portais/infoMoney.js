const puppeteer = require('puppeteer')
const { conectaBD, desconectaBD, insereNoticias } = require("./BD_handler")


async function coletaDadosInfoMoney(pagina, link) {
  await pagina.goto(link, { waitUntil: "domcontentloaded"})
  const noticia = await pagina.evaluate(() => {
    const dados = {
      portal: {nome: "InfoMoney"},
      _id: window.location.href
    }

    // Manchete
    let manchete = document.querySelector(".text-3xl")
    if(manchete) dados.manchete = manchete.textContent.trim()
    else return null
  
    // Lide
    let lide = document.querySelector(".text-lg")
    if(lide) dados.lide = lide.textContent.trim()


    // Datatime[itemprop="dateModified"]
    let dataPublicacao = document.querySelector("div[data-ds-component='author-small'] time")

    if(dataPublicacao) {
        dados.data = dataPublicacao.getAttribute("datetime")
    } else {
      return null
    }

    // Autores
    let autores = Array.from(document.querySelectorAll("div[data-ds-component='author-small'] a.text-base, div[data-ds-component='author-small'] span.text-base")).map(x => x.textContent.trim())
    dados.autor = autores.map(x => {
      let autor = {nome: x}
      return autor
    })
  
      // tag
    let tags = Array.from(document.querySelectorAll('[data-ds-component="related-topics"] a[aria-label]')).map(el => el.getAttribute("aria-label"))
     
    if(tags && tags.length > 0) dados.assunto = tags.map(x => {
      const tag = {}
      tag.TAG = x.trim().toUpperCase()
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

async function scrapInfoMoney(URL, secao) {
  const browser = await puppeteer.launch({headless:true})
  const scrapingPage = await browser.newPage()
  const paginaPortal = await browser.newPage()
  await paginaPortal.goto(`${URL}`, { waitUntil: "domcontentloaded" })

  const {db, client} = await conectaBD()
  let dict = []

  try{

    for(let i = 1; i <= 1000; i++){
        await paginaPortal.bringToFront()
        await paginaPortal.evaluate(() => {
          window.scrollTo(0, document.body.scrollHeight);
        });

        let links = await paginaPortal.evaluate(() => {
          return Array.from(document.querySelectorAll(".size-28 a")).map(el => el.getAttribute("href"))
        })
        
        // await new Promise(resolve => setTimeout(resolve, 4000))
        await paginaPortal.evaluate(() => {
          const artigosAntigos = document.querySelectorAll("[data-ds-commponent='card-infoproduct-lg'], [data-ds-component='card-sm'], [data-ds-component='card-lg'], [data-ds-component='card-xl']");
          artigosAntigos.forEach(artigo => artigo.remove());
        });
        await new Promise(resolve => setTimeout(resolve, 1000))
        
        try {
          let clickResult = await paginaPortal.locator('[data-ds-component="load-more-cards"] button.flex').click({count: 2 ,delay: 1000})
        } catch (e) {
          return null
          console.log("Não foi possível carregar novos conteúdos")
          console.log(e)
        }   
        // await new Promise(resolve => setTimeout(resolve, 4000))
        
        await scrapingPage.bringToFront()
        for (let i = 0; i < links.length; i++) {
          let temp = await coletaDadosInfoMoney(scrapingPage, links[i])
  
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
  } catch (err) {
    return
    console.error("Erro:", err)
  } finally {
    await insereNoticias(dict, db)
    await scrapingPage.close()
    await desconectaBD(client)
    await browser.close()
  }	
}


async function scrapingInfoMoney(){  
  console.log('Coletando InfoMoney...')
  scrapInfoMoney("https://www.infomoney.com.br/economia/", "Economia")
  
}

module.exports = {scrapingInfoMoney}