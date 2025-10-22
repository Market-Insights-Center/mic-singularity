# --- Imports for web_command ---
import asyncio
import os
import re
import json
import uuid
from typing import List, Dict, Optional

import yfinance as yf
import networkx as nx
import matplotlib.pyplot as plt
from tabulate import tabulate
import humanize

# --- Helper Functions (copied or moved for self-containment) ---

def ask_singularity_input(prompt: str, validation_fn=None, error_msg="Invalid input.", default_val=None) -> Optional[str]:
    """Helper function to ask for user input with optional validation."""
    while True:
        full_prompt = f"{prompt}"
        if default_val is not None:
            full_prompt += f" (default: {default_val}, press Enter)"
        full_prompt += ": "
        user_response = input(full_prompt).strip()
        if not user_response and default_val is not None:
            return str(default_val)
        if validation_fn:
            if validation_fn(user_response):
                return user_response
            else:
                print(error_msg)
        elif user_response:
            return user_response

def load_connections_from_js_for_web(filepath="database.js") -> Optional[list]:
    """Reads and parses the connectionsData array from the database.js file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        match = re.search(r'=\s*(\[.*\]);', content, re.DOTALL)
        if not match: return None
        json_string = re.sub(r',\s*([\]}])', r'\1', match.group(1))
        return json.loads(json_string)
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        return None

def find_company_data_for_web(identifier: str, connections_data: list) -> Optional[dict]:
    """Finds a company's data by name or ticker from the connections list."""
    search_term = identifier.lower()
    for conn in connections_data:
        if conn['Company Name'].lower() == search_term or (conn.get('Stock Ticker', '').lower() == search_term):
            return {'name': conn['Company Name'], 'ticker': conn.get('Stock Ticker')}
        if conn['Connecting To'].lower() == search_term or (conn.get('Connection Ticker', '').lower() == search_term):
            return {'name': conn['Connecting To'], 'ticker': conn.get('Connection Ticker')}
    return None

async def fetch_market_caps_for_web(tickers: set) -> dict:
    """Fetches market cap data for a given set of tickers."""
    if not tickers: return {}
    market_caps = {}
    try:
        ticker_data = await asyncio.to_thread(yf.Tickers, list(tickers))
        async def get_cap(ticker_obj):
            try:
                info = await asyncio.to_thread(lambda: ticker_obj.info)
                if cap := info.get('marketCap'):
                    return ticker_obj.ticker, cap
            except Exception: pass
            return None, None
        results = await asyncio.gather(*[get_cap(obj) for obj in ticker_data.tickers.values()])
        for ticker, cap in results:
            if ticker and cap: market_caps[ticker.upper()] = cap
    except Exception: pass
    return market_caps

async def generate_mic_web_graph(nodes, edges, display_mode, start_company_name):
    """Generates and saves a network graph image."""
    print("   -> Generating network graph image...")
    G = nx.DiGraph()
    node_to_level = {}
    for node_data in nodes:
        G.add_node(node_data['id'], level=node_data.get('level', 0))
        if node_data.get('level') is not None: node_to_level[node_data['id']] = node_data['level']
    for edge_data in edges:
        G.add_edge(edge_data['from'], edge_data['to'], label=edge_data.get('label', ''))
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(20, 16))
    pos = nx.multipartite_layout(G, subset_key="level") if 'tiered' in display_mode and node_to_level else nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, edge_color='gray', font_size=8, font_color='white', ax=ax)
    if 'direct' in display_mode:
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='blue', font_size=8, ax=ax)
    
    ax.set_title(f"M.I.C. Web Network for '{start_company_name}' ({display_mode})", fontsize=20, color='white')
    filename = f"mic_web_graph_{start_company_name.replace(' ','_')}_{uuid.uuid4().hex[:6]}.png"
    plt.savefig(filename, facecolor='black', bbox_inches='tight')
    plt.close(fig)
    print(f"   -> ✅ Graph saved as: {filename}")

# --- Main Command Handler ---

async def handle_web_command(args: List[str], is_called_by_ai: bool = False):
    """
    Replicates the functionality of the MICWEB project within the Singularity CLI.
    Generates a text-based report and a network graph image.
    """
    print("\n--- M.I.C. WEB Command ---")
    
    # 1. Load Data
    connections_data = load_connections_from_js_for_web()
    if not connections_data:
        print("-> Could not load connection data. Aborting.")
        return

    # 2. Get User Inputs
    company_input_str = ask_singularity_input("Enter a company name, ticker, or 'all'")
    if not company_input_str: return

    display_mode = ask_singularity_input("Select display mode (1:web-grouped, 2:web-direct, 3:tiered-grouped, 4:tiered-direct)", 
                                         validation_fn=lambda x: x in ['1','2','3','4'], default_val='1')
    display_mode_map = {'1': 'web-grouped', '2': 'web-direct', '3': 'tiered-grouped', '4': 'tiered-direct'}
    display_mode = display_mode_map[display_mode]

    max_level = float('inf')
    if company_input_str.lower() != 'all':
        stage_level = ask_singularity_input("Filter by Stage (1-5, or 'all')", default_val='all')
        if stage_level.isdigit() and 1 <= int(stage_level) <= 5:
            max_level = int(stage_level)
    
    fetch_caps = ask_singularity_input("Fetch and display market caps? (yes/no)", default_val='no').lower() == 'yes'

    # 3. Find Start Company and Traverse Connections (BFS)
    start_company = find_company_data_for_web(company_input_str, connections_data)
    if company_input_str.lower() != 'all' and not start_company:
        print(f"-> Company or ticker '{company_input_str}' not found.")
        return
        
    visited_levels = {}
    if company_input_str.lower() != 'all':
        start_node_name = start_company['name']
        queue = [(start_node_name, 0)]
        visited_levels[start_node_name] = 0

        head = 0
        while head < len(queue):
            current_name, current_level = queue[head]
            head += 1
            
            if current_level >= max_level:
                continue

            for conn in connections_data:
                from_name, to_name = conn['Company Name'], conn['Connecting To']
                neighbor = None
                if from_name == current_name: neighbor = to_name
                elif to_name == current_name: neighbor = from_name
                
                if neighbor and neighbor not in visited_levels:
                    visited_levels[neighbor] = current_level + 1
                    queue.append((neighbor, current_level + 1))

    # 4. Filter Connections and Prepare Graph Data
    connections_to_draw = connections_data if company_input_str.lower() == 'all' else [
        c for c in connections_data if c['Company Name'] in visited_levels and c['Connecting To'] in visited_levels
    ]
    
    nodes_for_graph, edges_for_graph, added_nodes = [], [], set()
    
    for conn in connections_to_draw:
        from_node, to_node = conn['Company Name'], conn['Connecting To']
        attr_node = conn['Connecting Attribute']
        
        for name in [from_node, to_node]:
            if name not in added_nodes:
                level = visited_levels.get(name)
                nodes_for_graph.append({'id': name, 'label': name, 'level': level, 'shape': 'box'})
                added_nodes.add(name)
        
        if 'direct' in display_mode:
            edges_for_graph.append({'from': from_node, 'to': to_node, 'label': attr_node})
        else: # Grouped mode
            if attr_node not in added_nodes:
                attr_level = None
                if visited_levels:
                    from_lvl, to_lvl = visited_levels.get(from_node), visited_levels.get(to_node)
                    if from_lvl is not None and to_lvl is not None:
                        attr_level = min(from_lvl, to_lvl) + 0.5
                nodes_for_graph.append({'id': attr_node, 'label': attr_node, 'shape': 'ellipse', 'level': attr_level})
                added_nodes.add(attr_node)
            edges_for_graph.append({'from': from_node, 'to': attr_node})
            edges_for_graph.append({'from': attr_node, 'to': to_node})
            
    # 5. Market Cap Analysis
    if fetch_caps and company_input_str.lower() != 'all':
        print("\n--- Market Cap Analysis ---")
        print("-> Fetching market cap data...")
        
        name_to_ticker_map = {}
        all_tickers = set()
        for c in connections_data:
            if c.get("Stock Ticker"): 
                name_to_ticker_map[c['Company Name']] = c["Stock Ticker"]
                all_tickers.add(c["Stock Ticker"])
            if c.get("Connection Ticker"): 
                name_to_ticker_map[c['Connecting To']] = c["Connection Ticker"]
                all_tickers.add(c["Connection Ticker"])
        
        market_caps_data = await fetch_market_caps_for_web(all_tickers)
        
        stage_caps = [0] * (int(max_level) + 2 if max_level != float('inf') else 6)
        processed_tickers = set()
        
        start_ticker = start_company.get('ticker')
        start_cap = market_caps_data.get(start_ticker.upper()) if start_ticker else None
        
        print(f"\nStart Company: {start_company['name']} ({start_ticker or 'N/A'})")
        print(f"  -> Market Cap: {humanize.intword(start_cap) if start_cap else 'N/A'}")
        
        for node in nodes_for_graph:
            if node['shape'] != 'box': continue
            level = node.get('level')
            ticker = name_to_ticker_map.get(node['id'])
            if ticker and ticker not in processed_tickers and level is not None:
                cap = market_caps_data.get(ticker.upper())
                if cap and level < len(stage_caps):
                    stage_caps[level] += cap
                processed_tickers.add(ticker)

        cumulative_cap = stage_caps[0] if stage_caps else 0
        last_stage_to_show = int(max_level) if max_level != float('inf') else 5
        
        print("\nCumulative Market Cap by Stage:")
        for i in range(1, last_stage_to_show + 1):
            if i < len(stage_caps):
                cumulative_cap += stage_caps[i]
                print(f"  -> Stage {i} Cumulative: {humanize.intword(cumulative_cap)}")
    
    # 6. Generate Text Report of Connections
    print("\n--- Connection Report ---")
    connections_to_draw.sort(key=lambda x: (visited_levels.get(x['Company Name'], 99), x['Company Name']))
    connection_table = [[
        conn['Company Name'], 
        f"Lvl {visited_levels.get(conn['Company Name'])}" if conn['Company Name'] in visited_levels else 'N/A',
        conn['Connecting Attribute'], 
        conn['Connecting To'],
        f"Lvl {visited_levels.get(conn['Connecting To'])}" if conn['Connecting To'] in visited_levels else 'N/A',
    ] for conn in connections_to_draw]
    print(tabulate(connection_table, headers=["From", "Stage", "Connection", "To", "Stage"], tablefmt="grid"))
    
    # 7. Generate Graph Image
    await generate_mic_web_graph(nodes_for_graph, edges_for_graph, display_mode, company_input_str)