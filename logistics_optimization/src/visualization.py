import folium


def draw_logistics_map(
    warehouses_coords, shops_coords, optimal_routes, filename="logistics_map.html"
):
    """
    Рисует интерактивную карту с результатами оптимизации логистики.
    """
    m = folium.Map(location=[59.9400, 30.2400], zoom_start=12)

    warehouse_names = ["Парнас (Север)", "Шушары (Юг)", "Янино (Восток)"]
    warehouse_colors = ["red", "blue", "green"]

    for i, w_coord in enumerate(warehouses_coords):
        folium.Marker(
            location=w_coord,
            popup=f"Склад: {warehouse_names[i]}",
            icon=folium.Icon(color=warehouse_colors[i], icon="info-sign"),
        ).add_to(m)

    for j, s_coord in enumerate(shops_coords):
        folium.CircleMarker(
            location=s_coord,
            radius=5,
            popup=f"Магазин №{j + 1}",
            color="gray",
            fill=True,
            fill_opacity=0.7,
        ).add_to(m)

    num_warehouses, num_shops = optimal_routes.shape
    for i in range(num_warehouses):
        for j in range(num_shops):
            volume = optimal_routes[i, j]
            if volume > 0.5:
                weight = max(1, volume / 10)

                folium.PolyLine(
                    locations=[warehouses_coords[i], shops_coords[j]],
                    color=warehouse_colors[i],
                    weight=weight,
                    opacity=0.6,
                    tooltip=f"Везем {volume:.1f} паллет",
                ).add_to(m)

    m.save(filename)
    print(f"\nКарта успешно сохранена в файл: {filename}")
    print("Открой этот файл в любом браузере!")
