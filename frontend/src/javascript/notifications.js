 import { showMessage } from './script.js'; // Assumindo que showMessage é exportado

        document.addEventListener('DOMContentLoaded', async () => {
            const form = document.getElementById('notificationPreferencesForm');
            const keywordsInput = document.getElementById('keywordsInput');
            const messageElement = document.getElementById('message');
            const currentKeywordsDisplay = document.getElementById('currentKeywordsDisplay');

            // Variável para armazenar as palavras-chave carregadas atualmente
            let loadedKeywords = []; 

            const userName = localStorage.getItem('user_name'); // O nome do usuário logado
            const jwtToken = localStorage.getItem('jwt_token');

            if (!jwtToken || !userName) {
                showMessage(messageElement, 'Você precisa estar logado para acessar as preferências de notificação.', 'error');
                setTimeout(() => { window.location.href = 'login.html'; }, 2000);
                return;
            }

            // Função para renderizar as palavras-chave na div de exibição
            function renderKeywords(keywords) {
                currentKeywordsDisplay.innerHTML = ''; // Limpa o conteúdo existente
                if (keywords && keywords.length > 0) {
                    keywords.forEach(keyword => {
                        const keywordTag = document.createElement('span');
                        keywordTag.classList.add('keyword-tag');
                        keywordTag.textContent = keyword;
                        
                        // Botão de remover (funcionalidade completa agora)
                        const removeButton = document.createElement('span');
                        removeButton.classList.add('remove-tag');
                        removeButton.innerHTML = '&times;'; // Caractere 'x'
                        removeButton.onclick = () => removeKeyword(keyword); // Adicionar evento de clique para remover
                        keywordTag.appendChild(removeButton);

                        currentKeywordsDisplay.appendChild(keywordTag);
                    });
                } else {
                    currentKeywordsDisplay.innerHTML = '<p class="no-keywords-message">Nenhuma palavra-chave adicionada ainda.</p>';
                }
            }

            // Função para carregar as preferências atuais do usuário (e popular loadedKeywords)
            async function loadPreferences() {
                try {
                    currentKeywordsDisplay.innerHTML = 'Carregando palavras-chave...';
                    const response = await fetch('http://localhost:3000/api/user/preferences', {
                        method: 'GET',
                        headers: {
                            'Content-Type': 'application/json',
                            'Authorization': `Bearer ${jwtToken}`
                        }
                    });
                    const data = await response.json();

                    if (response.ok && data.preferences) {
                        loadedKeywords = data.preferences.keywords || []; // Armazena as palavras-chave
                        renderKeywords(loadedKeywords); // Renderiza a lista completa
                        // Não preenche o keywordsInput com todas as palavras-chave, pois ele é para "adicionar"
                    } else {
                        showMessage(messageElement, data.msg || 'Não foi possível carregar as preferências.', 'error');
                        renderKeywords([]);
                    }
                } catch (error) {
                    console.error('Erro ao carregar preferências:', error);
                    showMessage(messageElement, 'Erro ao conectar com o servidor para carregar preferências.', 'error');
                    renderKeywords([]);
                }
            }

            // Função para remover uma palavra-chave
            async function removeKeyword(keywordToRemove) {
                if (!confirm(`Tem certeza que deseja remover a palavra-chave "${keywordToRemove}"?`)) {
                    return;
                }

                // Cria uma nova lista sem a palavra-chave a ser removida
                const updatedKeywords = loadedKeywords.filter(kw => kw !== keywordToRemove);

                await saveKeywordsToBackend(updatedKeywords, `Palavra-chave "${keywordToRemove}" removida com sucesso!`);
            }

            // Função auxiliar para salvar a lista COMPLETA de palavras-chave no backend
            async function saveKeywordsToBackend(keywordsArray, successMessage) {
                try {
                    const response = await fetch('http://localhost:3000/api/user/preferences', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'Authorization': `Bearer ${jwtToken}`
                        },
                        body: JSON.stringify({ keywords: keywordsArray })
                    });
                    const data = await response.json();

                    if (response.ok) {
                        showMessage(messageElement, successMessage, 'success');
                        keywordsInput.value = ''; // Limpa o campo de entrada após salvar
                        loadPreferences(); // Recarrega a lista para atualizar a UI
                    } else {
                        showMessage(messageElement, data.msg || 'Erro ao salvar palavras-chave.', 'error');
                    }
                } catch (error) {
                    console.error('Erro ao salvar palavras-chave:', error);
                    showMessage(messageElement, 'Erro ao conectar com o servidor para salvar palavras-chave.', 'error');
                }
            }


            // Lidar com o envio do formulário (para ADICIONAR novas palavras-chave)
            if (form) {
                form.addEventListener('submit', async (e) => {
                    e.preventDefault();

                    const newKeywordsRaw = keywordsInput.value;
                    const newKeywordsArray = newKeywordsRaw.split(',').map(kw => kw.trim().toLowerCase()).filter(kw => kw.length > 0);
                    
                    if (!newKeywordsArray.length) {
                        showMessage(messageElement, 'Por favor, insira pelo menos uma palavra-chave.', 'error');
                        return;
                    }

                    // Combina as palavras-chave existentes com as novas, removendo duplicatas
                    const combinedUniqueKeywords = Array.from(new Set([...loadedKeywords, ...newKeywordsArray]));

                    await saveKeywordsToBackend(combinedUniqueKeywords, 'Palavras-chave adicionadas com sucesso!');
                });
            }

            loadPreferences(); // Carrega as palavras-chave ao carregar a página
        });
