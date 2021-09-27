let
    basicfilter = function(counts)
        markers  = (
            yolk = scRNA.locus(counts, "sisA", "CG8129", "Corp", "CG8195", "CNT1", "ZnT77C"),
            pole = scRNA.locus(counts, "pgc"),
            dvir = scRNA.searchloci(counts, "Dvir_")
        )

        counts = scRNA.filtercell(counts) do cell, _
            (sum(cell[markers.yolk]) < 10
          && sum(cell[markers.pole]) < 3
          && sum(cell[markers.dvir]) < .25*sum(cell))
        end

        counts = scRNA.filtergene(counts) do _, gene
            !occursin("Dvir_", gene)
        end

        return counts
    end

    Parameters("basic";
        subdir="rep",
        filter=basicfilter,
    )
end
