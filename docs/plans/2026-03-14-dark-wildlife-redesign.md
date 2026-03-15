# Zoo Audio AI - Dark Wildlife Redesign

## Overview

Restyle the Gradio chatbot UI to a dark mode wildlife theme inspired by Stitch
"Chatbot AI" designs. Keeps Gradio as the framework; all changes are CSS theming
and minor Python layout adjustments.

## Color Palette

### Dark surfaces (layered depth)
| Token            | Hex                        | Use                              |
|------------------|----------------------------|----------------------------------|
| `--bg`           | `#0d0f1a`                  | Page background                  |
| `--surface`      | `#161829`                  | Chat area, main cards            |
| `--surface-alt`  | `#1e2038`                  | Elevated elements, input bar     |
| `--surface-high` | `#272a45`                  | Hover states, table headers      |
| `--border`       | `rgba(255,255,255,0.06)`   | Subtle dividers                  |

### Accent colors (nighttime safari)
| Token              | Value                        | Use                            |
|--------------------|------------------------------|--------------------------------|
| `--primary`        | `#4ade80`                    | Main actions, send button      |
| `--primary-hover`  | `#22c55e`                    | Hover on primary               |
| `--primary-glow`   | `rgba(74,222,128,0.15)`      | Glow behind primary elements   |
| `--accent-gold`    | `#fbbf24`                    | Feedback badges, highlights    |
| `--accent-purple`  | `#a78bfa`                    | Model badges, secondary info   |
| `--text`           | `#e8e8ed`                    | Primary text                   |
| `--text-muted`     | `#6b7085`                    | Secondary/placeholder text     |
| `--error`          | `#f87171`                    | Error states                   |

## Typography

- **Headlines:** Manrope (700/800 weight, tight tracking)
- **Body/Labels:** Inter (400/500/600 weight)
- Both imported from Google Fonts

## Border Radii

- Cards/chat bubbles: `24px`
- Buttons/inputs: `9999px` (pill)
- Inner elements: `16px`

## Components

### Header (sticky, glassmorphism)
- `--surface` at 80% opacity + `backdrop-blur: 20px`
- Left: green icon + "Zoo Audio AI" in Manrope bold
- Right: status badges (model, state, feedback count)

### Chat Bubbles
- **User:** right-aligned, green bg, dark text, rounded-3xl, green glow shadow
- **Bot:** left-aligned, `--surface-alt` bg, light text, bot avatar left

### Input Bar (floating, pill-shaped)
- Centered at bottom, `max-width: 720px`
- Dark glass bg, rounded-full
- Green circular send button, mic icon button

### Audio Upload
- Dashed border, dark bg, green border on hover
- Compact, above the input bar

### Status Badges
- In the header, pill-shaped, monospace
- Model: lavender tint, Feedback: gold tint, State: green/gold

### Welcome State (empty chat)
- Bento hero card: gradient surface-alt to purple
- Title + subtitle + faint waveform decoration
- Quick action pills: `/help`, `/stats`, `/versions`, `/play`
- Giant "ZOO" watermark bottom-right at 2% opacity

## Files Modified

- `zoo_audio/app.py` (CSS overhaul, layout changes, welcome state)
