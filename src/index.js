#!/usr/bin/env node

import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import dotenv from 'dotenv';
import { generateText, streamText } from 'ai';
import { createAnthropic } from '@ai-sdk/anthropic';
import { createOpenAI } from '@ai-sdk/openai';
import { createGoogleGenerativeAI } from '@ai-sdk/google';
import { ColorConsole } from './console.js';

global.console = new ColorConsole({
    stdout: process.stdout,
    stderr: process.stderr,
    timestamp: process.env.NODE_ENV !== 'production',
});

dotenv.config();

const PORT = process.env.PORT || 11434;

// Initialize providers based on available API keys
const providers = {};
if (process.env.ANTHROPIC_API_KEY) {
    providers.anthropic = createAnthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
}
if (process.env.OPENAI_API_KEY) {
    providers.openai = createOpenAI({ apiKey: process.env.OPENAI_API_KEY });
}
if (process.env.GEMINI_API_KEY) {
    providers.google = createGoogleGenerativeAI({ apiKey: process.env.GEMINI_API_KEY });
}
if (process.env.OPENROUTER_API_KEY) {
    providers.openrouter = createOpenAI({
        apiKey: process.env.OPENROUTER_API_KEY,
        baseURL: 'https://openrouter.ai/api/v1',
        compatibility: 'compatible',
        name: 'openrouter',
    });
}

if (Object.keys(providers).length === 0) {
    console.error('❌ No API keys found. Set ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, or OPENROUTER_API_KEY');
    process.exit(1);
}

const DEFAULT_MODELS_PATH = path.join(process.cwd(), 'models.json');

let models = {};
try {
    if (fs.existsSync(DEFAULT_MODELS_PATH)) {
        models = JSON.parse(fs.readFileSync(DEFAULT_MODELS_PATH, 'utf8'));
        console.log(`✅ Loaded models from ${DEFAULT_MODELS_PATH}`);
    } else {
        // Built-in models
        models = {
            'claude-3-5-sonnet': { provider: 'anthropic', model: 'claude-3-5-sonnet-latest' },
            'claude-4-sonnet': { provider: 'anthropic', model: 'claude-sonnet-4-latest' },
            'claude-3-haiku':    { provider: 'anthropic', model: 'claude-3-haiku-latest' },

            'gpt-5-mini': { provider: 'openai', model: 'gpt-5-mini' },
            'gpt-5-nano': { provider: 'openai', model: 'gpt-5-nano' },
            'gpt-4o-mini': { provider: 'openai', model: 'gpt-4o-mini' },
            'gpt-4.1-mini': { provider: 'openai', model: 'gpt-4.1-mini' },
            'gpt-4.1-nano': { provider: 'openai', model: 'gpt-4.1-nano' },

            'gemini-2.5-flash': { provider: 'google', model: 'gemini-2.5-flash' },
            'gemini-2.5-flash-lite': { provider: 'google', model: 'gemini-2.5-flash-lite' },

            'deepseek-r1': { provider: 'openrouter', model: 'deepseek/deepseek-r1-0528:free' },
        };
        console.log('ℹ️ Using built-in models. Create a models.json file to customize.');
    }
} catch (error) {
    console.error(`❌ Error loading models.json: ${error.message}`);
    process.exit(1);
}

// Utility functions
const getBody = request => new Promise(resolve => {
    let body = '';
    request.on('data', chunk => body += chunk);
    request.on('end', () => resolve(body ? JSON.parse(body) : {}));
});

const sendJSON = (response, data, status = 200) =>
    response.writeHead(status, {
        'Content-Type': 'application/json',
        'charset': 'utf-8',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type',
    }).end(JSON.stringify(data));


const validateModel = name => {
    const config = models[name];
    if (!config) {
        throw new Error(`Model ${name} not supported`);
    }
    if (!providers[config.provider]) {
        throw new Error(`Provider ${config.provider} not available`);
    }
    return config;
};

// Accepts: content as string OR array.
// - If array of strings → join into one string
// - If array with objects (parts) → pass through, just trim text parts
const prepareMessages = (msgs = []) => msgs
  .map(({ role, content }) => {
    if (Array.isArray(content)) {
      content = content.every(x => typeof x === 'string')
        ? content.map(s => s.trim()).filter(Boolean).join('\n')
        : content
            .map(p => (p?.type === 'text'
              ? { ...p, text: String(p.text || '').trim() }
              : p))
            .filter(p => (p?.type === 'text' ? p.text : p));
    } else if (typeof content === 'string') {
      content = content.trim();
    }

    const r = role === 'assistant' ? 'assistant'
            : role === 'system'   ? 'system'
            : 'user';

    return content && (Array.isArray(content) ? content.length : content)
      ? { role: r, content }
      : null;
  })
  .filter(Boolean);


// Generate complete text response using AI SDK
const generateResponse = async (modelConfig, messages, options = {}) => {
    const provider = providers[modelConfig.provider];
    const model = provider(modelConfig.model);

    const validMessages = prepareMessages(messages);

    if (validMessages.length === 0) {
        throw new Error('No valid messages found');
    }

    const result = await generateText({
        model,
        messages: validMessages,
        temperature: options.temperature,
        maxTokens: options.num_predict || 32768,
        topP: options.top_p,
    });

    // Handle different response formats from upstream
    let text = result.text;
    let reasoning = result.reasoning || null;
    let responseMessages = null;

    // If upstream returns messages array, extract the assistant's response
    if (result.messages && Array.isArray(result.messages)) {
        responseMessages = result.messages;

        // Find the last assistant message for the main response
        const assistantMessage = result.messages
            .filter(msg => msg.role === 'assistant')
            .pop();

        if (assistantMessage) {
            text = assistantMessage.content || assistantMessage.text || text;

            // Check if the assistant message has reasoning
            if (assistantMessage.reasoning) {
                reasoning = assistantMessage.reasoning;
            }
        }
    }

    // Return structured response
    return {
        text: text || '',
        reasoning: reasoning,
        messages: responseMessages,
    };
};

// Stream text response using AI SDK
const streamResponse = async (response, modelConfig, messages, options = {}, responseKey = 'message') => {
    const provider = providers[modelConfig.provider];
    const model = provider(modelConfig.model);

    const validMessages = prepareMessages(messages);

    if (validMessages.length === 0) {
        throw new Error('No valid messages found');
    }

    // Set streaming headers
    response.writeHead(200, {
        'Content-Type': 'application/x-ndjson',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
    });

    try {
        const result = await streamText({
            model,
            messages: validMessages,
            temperature: options.temperature,
            maxTokens: options.num_predict,
            topP: options.top_p,
        });

        const modelName = modelConfig.model;

        // Stream the tokens
        for await (const delta of result.textStream) {
            const chunk = {
                model: modelName,
                created_at: new Date().toISOString(),
                done: false,
            };

            // Set the content based on response type
            if (responseKey === 'message') {
                chunk.message = {
                    role: 'assistant',
                    content: delta,
                };
            } else if (responseKey === 'response') {
                chunk.response = delta;
            }

            // Send chunk as NDJSON
            response.write(JSON.stringify(chunk) + '\n');
        }

        // Send final chunk with done: true
        const finalChunk = {
            model: modelName,
            created_at: new Date().toISOString(),
            done: true,
        };

        // Set empty content for final chunk
        if (responseKey === 'message') {
            finalChunk.message = {
                role: 'assistant',
                content: '',
            };
        } else if (responseKey === 'response') {
            finalChunk.response = '';
        }

        // Add reasoning if available
        if (result.reasoning) {
            if (responseKey === 'message') {
                finalChunk.message.reasoning = result.reasoning;
            } else {
                finalChunk.reasoning = result.reasoning;
            }
        }

        response.write(JSON.stringify(finalChunk) + '\n');
        response.end();

    } catch (error) {
        console.error('Streaming error:', error.message);

        // Send error chunk
        const errorChunk = {
            model: modelConfig.model,
            created_at: new Date().toISOString(),
            done: true,
            error: error.message,
        };

        response.write(JSON.stringify(errorChunk) + '\n');
        response.end();
    }
};

// Route handlers
const handleModelGenerationRequest = async (request, response, messageExtractor, responseKey) => {
    try {

        const body = await getBody(request);
        console.info(body);
        const { model, options = {}, stream = false } = body;

        const modelConfig = validateModel(model);
        const messages = messageExtractor(body) || [];

        console.debug(model, messages, { stream });

        // Handle streaming vs non-streaming responses
        if (stream) {
            await streamResponse(response, modelConfig, messages, options, responseKey);
        } else {
            const result = await generateResponse(modelConfig, messages, options);

            const responseData = {
                model,
                created_at: new Date().toISOString(),
                done: true,
            };

            // Set the main content key based on the request type
            if (responseKey === 'message') {
                responseData.message = {
                    role: 'assistant',
                    content: result.text,
                };
            } else if (responseKey === 'response') {
                responseData.response = result.text;
            }

            // Add reasoning if available
            if (result.reasoning) {
                if (responseKey === 'message') {
                    responseData.message.reasoning = result.reasoning;
                } else {
                    responseData.reasoning = result.reasoning;
                }
            }

            // Add messages array if available (for debugging or advanced use cases)
            if (result.messages) {
                responseData.messages = result.messages;
            }

            sendJSON(response, responseData);
        }
    } catch (error) {
        console.error('API request error:', error.message);

        // If response hasn't been sent yet, send JSON error
        if (!response.headersSent) {
            sendJSON(response, { error: error.message }, 500);
        } else {
            // If streaming has started, send error chunk
            const errorChunk = {
                model: 'unknown',
                created_at: new Date().toISOString(),
                done: true,
                error: error.message,
            };
            response.write(JSON.stringify(errorChunk) + '\n');
            response.end();
        }
    }
};

const handleOpenAIChatCompletions = async (request, response) => {
  try {
    const body = await getBody(request);
    const {
      model,                 // string (your alias, e.g. "gpt-4o-mini" or "claude-..."),
      messages = [],         // OpenAI-style [{role, content}]
      stream = false,
      temperature,
      top_p,
      max_tokens,            // OpenAI name
      // (you can read other OpenAI params here if you want)
    } = body;

    const modelConfig = validateModel(model);
    const options = {
      temperature,
      top_p,
      num_predict: max_tokens,  // map OpenAI → your SDK option
    };

    // Non-streaming (simple JSON)
    if (!stream) {
      const result = await generateResponse(modelConfig, messages, options);
      const now = Math.floor(Date.now() / 1000);

      const payload = {
        id: `chatcmpl_${Math.random().toString(36).slice(2)}`,
        object: 'chat.completion',
        created: now,
        model,
        choices: [{
          index: 0,
          message: {
            role: 'assistant',
            content: result.text || '',
            ...(result.reasoning ? { reasoning: result.reasoning } : {})
          },
          finish_reason: 'stop'
        }],
        usage: {
          prompt_tokens: null,
          completion_tokens: null,
          total_tokens: null
        }
      };

      return sendJSON(response, payload);
    }

    // Streaming (Server-Sent Events, OpenAI-compatible)
    const provider = providers[modelConfig.provider];
    const sdkModel = provider(modelConfig.model);
    const validMessages = prepareMessages(messages);

    response.writeHead(200, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache, no-transform',
      'Connection': 'keep-alive',
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    });

    const now = Math.floor(Date.now() / 1000);
    const id = `chatcmpl_${Math.random().toString(36).slice(2)}`;

    const sendSSE = obj => response.write(`data: ${JSON.stringify(obj)}\n\n`);

    // First delta: role=assistant (OpenAI does this)
    sendSSE({
      id,
      object: 'chat.completion.chunk',
      created: now,
      model,
      choices: [{ index: 0, delta: { role: 'assistant' }, finish_reason: null }]
    });

    try {
      const result = await streamText({
        model: sdkModel,
        messages: validMessages,
        temperature,
        maxTokens: max_tokens,
        topP: top_p,
      });

      for await (const delta of result.textStream) {
        sendSSE({
          id,
          object: 'chat.completion.chunk',
          created: Math.floor(Date.now() / 1000),
          model,
          choices: [{ index: 0, delta: { content: delta }, finish_reason: null }]
        });
      }

      // Final chunk (finish_reason=stop)
      sendSSE({
        id,
        object: 'chat.completion.chunk',
        created: Math.floor(Date.now() / 1000),
        model,
        choices: [{ index: 0, delta: {}, finish_reason: 'stop' }]
      });

      // Stream terminator
      response.write('data: [DONE]\n\n');
      response.end();

    } catch (err) {
      // Error chunk (OpenAI-style)
      sendSSE({
        id,
        object: 'chat.completion.chunk',
        created: Math.floor(Date.now() / 1000),
        model,
        choices: [{ index: 0, delta: {}, finish_reason: 'error' }],
        error: { message: err.message }
      });
      response.write('data: [DONE]\n\n');
      response.end();
    }

  } catch (error) {
    if (!response.headersSent) {
      sendJSON(response, { error: error.message }, 500);
    } else {
      response.end();
    }
  }
};


const routes = {
    'GET /': (request, response) => response.end('Ollama is running in proxy mode.'),
    'GET /api/version': (request, response) => sendJSON(response, { version: '1.0.1e' }),
    'GET /api/tags': (request, response) => {
        const availableModels = Object.entries(models)
            .filter(([name, config]) => providers[config.provider])
            .map(([name]) => ({
                name,
                model: name,
                modified_at: new Date().toISOString(),
                size: 1000000000,
                digest: `sha256:${name.replace(/[^a-zA-Z0-9]/g, '')}`,
            }));
        sendJSON(response, { models: availableModels });
    },
    'POST /api/chat': async (request, response) => {
        await handleModelGenerationRequest(
            request,
            response,
            body => body.messages,
            'message',
        );
    },
    'POST /api/generate': async (request, response) => {
        await handleModelGenerationRequest(
            request,
            response,
            body => [{ role: 'user', content: body.prompt }],
            'response',
        );
    },
    // Roo Code requires OpenAI compatible endpoints.
    'POST /v1/chat/completions': async (req, res) => {
        await handleOpenAIChatCompletions(req, res);
    },
};

// HTTP Server
const ollamaProxyServer = http.createServer(async (request, response) => {
    const routeKey = `${request.method} ${request.url.split('?')[0]}`;


    console.info(routeKey);


    // Handle CORS preflight
    if (request.method === 'OPTIONS') {
        return response.writeHead(200, {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type',
        }).end();
    }

    try {
        const handler = routes[routeKey];
        if (handler) {
            await handler(request, response);
        } else {
            sendJSON(response, { error: 'Not found' }, 404);
        }
    } catch (error) {
        console.error('Server error:', error.message);
        if (!response.headersSent) {
            sendJSON(response, { error: 'Internal server error' }, 500);
        }
    }
});

// Start server
ollamaProxyServer.listen(PORT, () => {
    const availableModels = Object.keys(models).filter(name => providers[models[name].provider]);

    console.log(`🚀 Ollama Proxy with Streaming running on http://localhost:${PORT}`);
    console.log(`🔑 Providers: ${Object.keys(providers).join(', ')}`);
    console.log(`📋 Available models: ${availableModels.join(', ')}`);
});

// Graceful shutdown
['SIGINT', 'SIGTERM'].forEach(signal => {
    process.on(signal, () => {
        console.log(`\n🛑 Received ${signal}, shutting down...`);
        ollamaProxyServer.close(() => process.exit(0));
    });
});
