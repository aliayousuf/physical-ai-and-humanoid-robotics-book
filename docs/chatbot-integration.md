# Chatbot Integration

This document explains how to integrate the RAG chatbot with the Docusaurus documentation site.

## Overview

The RAG (Retrieval-Augmented Generation) chatbot is designed to provide contextual answers to questions about the Physical AI and Humanoid Robotics book content. The chatbot is implemented as a React component that can be embedded in any page of the Docusaurus site.

## Components

The chatbot consists of the following components:

- `RagChatbot` - Main chatbot component
- `ChatInterface` - Input and display interface
- `Message` - Individual message display
- `TextSelectionHandler` - Handles text selection on the page
- `api.ts` - API client for backend communication

## Integration with Docusaurus

### Adding the Chatbot to Layout

To add the chatbot to all pages, you can modify the Docusaurus layout component. The chatbot is designed to be persistent across page navigation.

### Text Selection Feature

The chatbot includes a text selection feature that allows users to select text on the page and ask questions specifically about that text. When text is selected, the chatbot switches to "Selected Text" mode.

### API Communication

The chatbot communicates with the backend API at `/api/v1/chat/` endpoints:

- `/session` - Create new chat sessions
- `/query` - General RAG queries
- `/selected-text-query` - Queries with selected text context
- `/session/{id}/history` - Get conversation history

## Configuration

The chatbot can be configured with the following props:

- `backendUrl` - URL of the backend API (default: `/api/v1`)

## Styling

The chatbot includes its own CSS file at `src/css/rag-chatbot.css` with responsive design and a clean interface that matches the documentation style.

## Features

- **Persistent across pages**: Maintains conversation context when navigating
- **Text selection**: Ask questions about selected text specifically
- **Source citations**: Responses include links to relevant book sections
- **Mode switching**: Toggle between general and selected text modes
- **Responsive design**: Works on mobile and desktop devices

## Troubleshooting

If the chatbot is not appearing on pages:

1. Verify that the backend service is running
2. Check that the API endpoints are accessible
3. Confirm that the component is properly imported and rendered in the layout